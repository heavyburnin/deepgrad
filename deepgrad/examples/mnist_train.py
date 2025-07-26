import os
import dill
import mmap
from array import array
from ctypes import c_float, c_int, POINTER, cast, memmove, addressof
from concurrent.futures import ThreadPoolExecutor
from tqdm import trange, tqdm
from deepgrad.tensor import Tensor
from deepgrad.model import MNISTConvNet
from deepgrad.optim import Adam

# --- Constants ---
INPUT_SIZE = 784
IMAGE_SHAPE = (1, 28, 28)
BATCH_SIZE = 512
PASSES = 25
TRAIN_BIN = 'deepgrad/examples/datasets/mnist_train.bin'
TEST_BIN = 'deepgrad/examples/datasets/mnist_test.bin'
TRAIN_CSV = 'deepgrad/examples/datasets/mnist_train.csv'
TEST_CSV = 'deepgrad/examples/datasets/mnist_test.csv'

# --- Data Utilities ---
def convert_csv_to_bin(csv_path, bin_path, normalize=False):
    MEAN, STD = 0.1307, 0.3081
    with open(csv_path, 'r') as fin, open(bin_path, 'wb') as fout:
        next(fin)  # skip header
        for line in fin:
            parts = line.strip().split(',')
            label = c_int(int(parts[0]))
            fout.write(label)
            pixels = (float(p)/255.0 for p in parts[1:])
            if normalize:
                pixels = ((p - MEAN) / STD for p in pixels)
            fout.write(array('f', pixels).tobytes())

def load_dataset(bin_path, input_size):
    with open(bin_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        sample_size = 1 + input_size
        total = mm.size() // (sample_size * 4)
        return mm, total, sample_size

def build_batch(mm, indices, input_size, x_buf, y_buf):
    sample_bytes = 4 + input_size * 4
    for i, idx in enumerate(indices):
        offset = idx * sample_bytes
        memmove(cast(addressof(y_buf) + i * 4, POINTER(c_int)), mm[offset:offset+4], 4)
        memmove(cast(addressof(x_buf) + i * input_size * 4, POINTER(c_float)),
                mm[offset+4:offset+sample_bytes], input_size * 4)
    return x_buf, y_buf

# --- Model Wrapper ---
class Model:
    def __init__(self): self.model = MNISTConvNet()
    def __call__(self, x): return self.model(x)
    def parameters(self): return self.model.parameters()
    def train(self): self.model.train()
    def eval(self): self.model.eval()

def accuracy(pred, target):
    pred_data, target_data = pred.data, target.data
    n, c = pred.shape
    correct = sum(max(range(c), key=lambda i: pred_data[j * c + i]) == target_data[j] for j in range(n))
    return correct / n

def evaluate(model, mm, total, batch_size, input_size, x_buf, y_buf):
    model.eval()
    correct, count = 0, 0
    for i in range(0, total, batch_size):
        indices = list(range(i, min(i + batch_size, total)))
        x_buf, y_buf = build_batch(mm, indices, input_size, x_buf, y_buf)
        x = Tensor.from_ctypes(x_buf, requires_grad=False, shape=(len(indices), *IMAGE_SHAPE), size=len(indices) * INPUT_SIZE)
        y = Tensor.from_ctypes(y_buf, requires_grad=False, shape=(len(indices),), size=len(indices))
        correct += accuracy(model(x.detach()), y) * len(indices)
        count += len(indices)
    return 100 * correct / count

# --- Main Training Loop ---
def main():
    if not os.path.exists(TRAIN_BIN):
        convert_csv_to_bin(TRAIN_CSV, TRAIN_BIN, normalize=True)
        convert_csv_to_bin(TEST_CSV, TEST_BIN, normalize=True)

    mm_train, n_train, _ = load_dataset(TRAIN_BIN, INPUT_SIZE)
    mm_test, n_test, _ = load_dataset(TEST_BIN, INPUT_SIZE)

    x_buf = (c_float * (BATCH_SIZE * INPUT_SIZE))()
    y_buf = (c_int * BATCH_SIZE)()
    x_buf_next = (c_float * (BATCH_SIZE * INPUT_SIZE))()
    y_buf_next = (c_int * BATCH_SIZE)()

    model = Model()
    model.train()
    opt = Adam(model.parameters(), lr=0.001, beta1=0.9, beta2=0.99)

    executor = ThreadPoolExecutor(max_workers=1)
    batches_per_epoch = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
    total_steps = PASSES * batches_per_epoch

    # Prefetch first batch
    indices = list(range(0, min(BATCH_SIZE, n_train)))
    future = executor.submit(build_batch, mm_train, indices, INPUT_SIZE, x_buf, y_buf)

    for step in trange(total_steps, desc="Training", ncols=100):
        batch_idx = step % batches_per_epoch
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_train)
        indices = list(range(start, end))

        # Get prefetched batch
        x_buf, y_buf = future.result()

        # Prefetch next batch
        next_idx = (batch_idx + 1) % batches_per_epoch
        next_indices = list(range(next_idx * BATCH_SIZE, min((next_idx + 1) * BATCH_SIZE, n_train)))
        future = executor.submit(build_batch, mm_train, next_indices, INPUT_SIZE, x_buf_next, y_buf_next)

        x = Tensor.from_ctypes(x_buf, requires_grad=True, shape=(len(indices), *IMAGE_SHAPE), size=len(indices) * INPUT_SIZE)
        y = Tensor.from_ctypes(y_buf, requires_grad=False, shape=(len(indices),), size=len(indices))

        opt.zero_grad_c()
        loss = model(x).cross_entropy(y, label_smoothing=0.05, use_label_smoothing=1)
        loss.backward()
        opt.step()
        loss.release_graph()

        # Swap buffers
        x_buf, x_buf_next = x_buf_next, x_buf
        y_buf, y_buf_next = y_buf_next, y_buf

        if (step + 1) % batches_per_epoch == 0:
            acc = evaluate(model, mm_test, n_test, BATCH_SIZE, INPUT_SIZE, x_buf, y_buf)
            tqdm.write(f"Epoch {(step + 1) // batches_per_epoch}: Test Accuracy = {acc:.2f}%")

    executor.shutdown()
    with open('deepgrad/examples/mnist_model.pkl', 'wb') as f:
        dill.dump(model, f)
    print("Model saved to mnist_model.pkl")

if __name__ == '__main__':
    main()
