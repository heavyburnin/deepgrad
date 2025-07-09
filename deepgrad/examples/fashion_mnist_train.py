import random
import os
import dill
import mmap
from tqdm import tqdm, trange
from array import array
from ctypes import c_float, POINTER, cast, memmove, addressof, c_int
from deepgrad.tensor import Tensor
from deepgrad.model import FashionConvNet
from deepgrad.optim import Adam, AdamW

# --- Dataset Conversion and Loader ---
def convert_csv_to_bin(csv_path, bin_path, normalize=False):
    FASHION_MNIST_MEAN = 0.2860406
    FASHION_MNIST_STD = 0.3530242
    with open(csv_path, 'r') as f_csv, open(bin_path, 'wb') as f_bin:
        next(f_csv)
        for line in f_csv:
            values = line.strip().split(',')
            label_idx = int(values[0])
            f_bin.write(c_int(label_idx))
            if normalize:
                image = array('f', ((float(px) / 255.0 - FASHION_MNIST_MEAN) / FASHION_MNIST_STD for px in values[1:]))
            else:
                image = array('f', (float(px) / 255.0 for px in values[1:]))
            f_bin.write(image.tobytes())

def load_bin_dataset(bin_path, num_samples, input_size):
    sample_size = input_size + 1
    with open(bin_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        actual_bytes = mm.size()
        expected_bytes = num_samples * sample_size * 4
        if actual_bytes < expected_bytes:
            num_samples = actual_bytes // (sample_size * 4)
        return mm, num_samples, sample_size

def build_batch_from_mmap(mm, sample_indices, input_size, x_array=None, y_array=None):
    sample_bytes = 4 + input_size * 4
    batch_size = len(sample_indices)
    total_input = batch_size * input_size

    if x_array is None:
        x_array = (c_float * total_input)()
    if y_array is None:
        y_array = (c_int * batch_size)()

    for i, sample_idx in enumerate(sample_indices):
        offset = sample_idx * sample_bytes
        label_ptr = cast(addressof(y_array) + i * 4, POINTER(c_int))
        memmove(label_ptr, mm[offset : offset + 4], 4)

        x_ptr = cast(addressof(x_array) + i * input_size * 4, POINTER(c_float))
        memmove(x_ptr, mm[offset + 4 : offset + sample_bytes], input_size * 4)

    return x_array, y_array

# --- Model Wrapper ---
class Model:
    def __init__(self):
        self.model = FashionConvNet()
    def __call__(self, x):
        return self.model(x)
    def parameters(self):
        return self.model.parameters()
    def train(self):
        self.model.train()
    def eval(self):
        self.model.eval()

# --- Evaluation function ---
def accuracy(pred, target):
    logits = pred.data
    targets = target.data
    batch_size = pred.shape[0]
    num_classes = pred.shape[1]
    correct = 0
    for j in range(batch_size):
        pred_index = max(range(num_classes), key=lambda i: logits[j * num_classes + i])
        if pred_index == targets[j]:
            correct += 1
    return correct / batch_size

def get_test_acc(model, mm_test, batch_size, input_size, num_samples, x_array, y_array):
    model.eval()
    correct = 0
    total = 0
    for i in range(0, num_samples, batch_size):
        actual_bs = min(batch_size, num_samples - i)
        indices = list(range(i, i + actual_bs))

        build_batch_from_mmap(mm_test, indices, input_size, x_array, y_array)

        x = Tensor(x_array, requires_grad=False, shape=(actual_bs, 1, 28, 28), size=actual_bs * input_size)
        y = Tensor(y_array, requires_grad=False, shape=(actual_bs,), size=actual_bs)

        pred = model(x)
        correct += accuracy(pred.detach(), y) * actual_bs
        total += actual_bs
    return (correct / total) * 100

def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        dill.dump(model, f)

# --- Main ---
INPUT_SIZE = 784
BATCH_SIZE = 512
PASSES = 75
NUM_TRAIN_SAMPLES = 60000
NUM_TEST_SAMPLES = 10000
x_array = (c_float * (BATCH_SIZE * INPUT_SIZE))()
y_array = (c_int * BATCH_SIZE)()

if __name__ == '__main__':
    train_bin = 'deepgrad/examples/datasets/fashion_mnist_train.bin'
    test_bin  = 'deepgrad/examples/datasets/fashion_mnist_test.bin'
    if not os.path.exists(train_bin) or not os.path.exists(test_bin):
        convert_csv_to_bin('deepgrad/examples/datasets/fashion_mnist_train.csv', train_bin, normalize=True)
        convert_csv_to_bin('deepgrad/examples/datasets/fashion_mnist_test.csv', test_bin, normalize=True)

    mm_train = load_bin_dataset(train_bin, NUM_TRAIN_SAMPLES, INPUT_SIZE)[0]
    mm_test = load_bin_dataset(test_bin, NUM_TEST_SAMPLES, INPUT_SIZE)[0]

    model = Model()
    model.train()
    total_batches = (NUM_TRAIN_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    total_steps = total_batches * PASSES

    opt = AdamW(model.parameters(), lr=0.001, beta1=0.9, beta2=0.99, weight_decay=0.0005)
    # opt = AdamW(model.parameters(), lr=0.001, beta1=0.9, beta2=0.99)
    # scheduler = LinearDecayLR(opt, start_lr=0.009, end_lr=0.004, total_steps=total_steps)

    for passes in trange(total_steps, desc="Training", ncols=100):
        batch_idx = passes % total_batches
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, NUM_TRAIN_SAMPLES)
        batch_indices = list(range(start, end))

        x_array, y_array = build_batch_from_mmap(mm_train, batch_indices, INPUT_SIZE, x_array, y_array)
        x = Tensor(x_array, requires_grad=True, shape=(len(batch_indices), 1, 28, 28), size=len(batch_indices) * INPUT_SIZE)
        y = Tensor(y_array, requires_grad=False, shape=(len(batch_indices),), size=len(batch_indices))

        out = model(x)
        # loss = out.cross_entropy(y)
        loss = out.cross_entropy(y, label_smoothing=0.1, use_label_smoothing=1)
        loss.backward()
        opt.step()
        opt.zero_grad_c()
        # scheduler.step()

        if (passes + 1) % total_batches == 0:
            epoch = (passes + 1) // total_batches
            acc = get_test_acc(model, mm_test, BATCH_SIZE, INPUT_SIZE, NUM_TEST_SAMPLES, x_array, y_array)
            tqdm.write(f"Pass {epoch} Test Accuracy: {acc:.2f}%")

    # Final test accuracy
    final_acc = get_test_acc(model, mm_test, BATCH_SIZE, INPUT_SIZE, NUM_TEST_SAMPLES, x_array, y_array)
    print(f"Final Test Accuracy: {final_acc:.2f}%")

    save_model(model, 'deepgrad/examples/fashion_model.pkl')
    print("Model saved to fashion_model.pkl")
