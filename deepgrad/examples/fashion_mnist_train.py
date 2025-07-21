import random
import os
import dill
import mmap
from tqdm import tqdm, trange
from array import array
from ctypes import c_float, c_int, POINTER, cast, memmove, addressof
from concurrent.futures import ThreadPoolExecutor
from deepgrad.tensor import Tensor
from deepgrad.model import FashionConvNet
from deepgrad.optim import AdamW

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
    
    # Pre-allocate arrays if not provided
    if x_array is None:
        x_array = (c_float * (batch_size * input_size))()
    if y_array is None:
        y_array = (c_int * batch_size)()
    
    # Use a single contiguous memory read per sample where possible
    for i, sample_idx in enumerate(sample_indices):
        offset = sample_idx * sample_bytes
        # Copy label
        label_ptr = cast(addressof(y_array) + i * 4, POINTER(c_int))
        memmove(label_ptr, mm[offset : offset + 4], 4)
        # Copy image data
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

        x_array, y_array = build_batch_from_mmap(mm_test, indices, input_size, x_array, y_array)

        x = Tensor.from_ctypes(x_array, requires_grad=False, shape=(actual_bs, 1, 28, 28), size=actual_bs * input_size)
        y = Tensor.from_ctypes(y_array, requires_grad=False, shape=(actual_bs,), size=actual_bs)

        pred = model(x.detach())
        correct += accuracy(pred.detach(), y) * actual_bs
        total += actual_bs
    return (correct / total) * 100

def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        dill.dump(model, f)

# --- Main ---
INPUT_SIZE = 784
BATCH_SIZE = 256
PASSES = 50
NUM_TRAIN_SAMPLES = 60000
NUM_TEST_SAMPLES = 10000
x_array = (c_float * (BATCH_SIZE * INPUT_SIZE))()
y_array = (c_int * BATCH_SIZE)()
x_array_next = (c_float * (BATCH_SIZE * INPUT_SIZE))()  # Buffer for next batch
y_array_next = (c_int * BATCH_SIZE)()  # Buffer for next batch

if __name__ == '__main__':
    train_bin = 'deepgrad/examples/datasets/fashion_mnist_train.bin'
    test_bin = 'deepgrad/examples/datasets/fashion_mnist_test.bin'
    if not os.path.exists(train_bin) or not os.path.exists(test_bin):
        convert_csv_to_bin('deepgrad/examples/datasets/fashion_mnist_train.csv', train_bin, normalize=True)
        convert_csv_to_bin('deepgrad/examples/datasets/fashion_mnist_test.csv', test_bin, normalize=True)

    mm_train, NUM_TRAIN_SAMPLES, _ = load_bin_dataset(train_bin, NUM_TRAIN_SAMPLES, INPUT_SIZE)
    mm_test, NUM_TEST_SAMPLES, _ = load_bin_dataset(test_bin, NUM_TEST_SAMPLES, INPUT_SIZE)

    model = Model()
    model.train()
    total_batches = (NUM_TRAIN_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    total_steps = total_batches * PASSES

    opt = AdamW(model.parameters(), lr=0.001, beta1=0.9, beta2=0.99, weight_decay=0.01)

    # Initialize thread pool for prefetching
    executor = ThreadPoolExecutor(max_workers=1)

    # Prefetch first batch
    batch_idx = 0
    start = batch_idx * BATCH_SIZE
    end = min(start + BATCH_SIZE, NUM_TRAIN_SAMPLES)
    batch_indices = list(range(start, end))
    future = executor.submit(build_batch_from_mmap, mm_train, batch_indices, INPUT_SIZE, x_array, y_array)

    for passes in trange(total_steps, desc="Training", ncols=100):
        batch_idx = passes % total_batches
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, NUM_TRAIN_SAMPLES)
        batch_indices = list(range(start, end))

        # Get current batch from previous prefetch
        x_array, y_array = future.result()

        # Prefetch next batch
        next_batch_idx = (batch_idx + 1) % total_batches
        next_start = next_batch_idx * BATCH_SIZE
        next_end = min(next_start + BATCH_SIZE, NUM_TRAIN_SAMPLES)
        next_batch_indices = list(range(next_start, next_end))
        future = executor.submit(build_batch_from_mmap, mm_train, next_batch_indices, INPUT_SIZE, x_array_next, y_array_next)

        # Process current batch
        x = Tensor.from_ctypes(x_array, requires_grad=True, shape=(len(batch_indices), 1, 28, 28), size=len(batch_indices) * INPUT_SIZE)
        y = Tensor.from_ctypes(y_array, requires_grad=False, shape=(len(batch_indices),), size=len(batch_indices))

        opt.zero_grad_c()
        out = model(x)
        loss = out.cross_entropy(y, label_smoothing=0.1, use_label_smoothing=1)
        loss.backward()
        opt.step()
        loss.release_graph()

        # Swap buffers for next iteration
        x_array, x_array_next = x_array_next, x_array
        y_array, y_array_next = y_array_next, y_array

        if (passes + 1) % total_batches == 0:
            epoch = (passes + 1) // total_batches
            acc = get_test_acc(model, mm_test, BATCH_SIZE, INPUT_SIZE, NUM_TEST_SAMPLES, x_array, y_array)
            tqdm.write(f"Pass {epoch} Test Accuracy: {acc:.2f}%")

    # Clean up
    executor.shutdown()
    save_model(model, 'deepgrad/examples/fashion_model.pkl')
    print("Model saved to fashion_model.pkl")