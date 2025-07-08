import random
import os
import dill
import mmap
from tqdm import trange
from array import array
from ctypes import c_float, POINTER, cast, memmove, addressof, c_int
from deepgrad.tensor import Tensor
from deepgrad.model import FashionConvNet
from deepgrad.optimizer import Adam

# --- Dataset Conversion and Loader ---
def convert_csv_to_bin(csv_path, bin_path):
    FASHION_MNIST_MEAN = 0.2860406
    FASHION_MNIST_STD = 0.3530242
    with open(csv_path, 'r') as f_csv, open(bin_path, 'wb') as f_bin:
        next(f_csv)
        for line in f_csv:
            values = line.strip().split(',')
            label_idx = int(values[0])
            f_bin.write(c_int(label_idx))
            image = array('f', ((float(px) / 255.0 - FASHION_MNIST_MEAN) / FASHION_MNIST_STD for px in values[1:]))
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

def build_batch_from_mmap(mm, sample_indices, input_size):
    sample_bytes = 4 + input_size * 4
    batch_size = len(sample_indices)
    total_input = batch_size * input_size
    x_array = (c_float * total_input)()
    y_array = (c_int * batch_size)()
    for i, sample_idx in enumerate(sample_indices):
        offset = sample_idx * sample_bytes
        label_ptr = cast(addressof(y_array) + i * 4, POINTER(c_int))
        memmove(label_ptr, mm[offset : offset + 4], 4)
        x_ptr = cast(addressof(x_array) + i * input_size * 4, POINTER(c_float))
        memmove(x_ptr, mm[offset + 4 : offset + sample_bytes], input_size * 4)
    return x_array, y_array

def random_batch(mm, batch_size, input_size, total_samples):
    indices = random.sample(range(total_samples), batch_size)
    bx, by = build_batch_from_mmap(mm, indices, input_size)
    x = Tensor(bx, requires_grad=True, shape=(batch_size, 1, 28, 28), size=batch_size * input_size)
    y = Tensor(by, requires_grad=False, shape=(batch_size,), size=batch_size)
    return x, y

def accuracy(pred, target):
    logits = pred.data
    targets = target.data
    num_classes = 10
    batch_size = len(targets)
    correct = 0
    for j in range(batch_size):
        pred_index = max(range(num_classes), key=lambda i: logits[j * num_classes + i])
        if pred_index == targets[j]:
            correct += 1
    return correct / batch_size

class OneCycleLR:
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3, div_factor=25.0, final_div_factor=1e4):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.step_num = 0

        self.initial_lr = self.max_lr / self.div_factor
        self.final_lr = self.max_lr / self.final_div_factor
        self.up_steps = int(self.total_steps * self.pct_start)
        self.down_steps = self.total_steps - self.up_steps

        self.optimizer.lr = self.initial_lr  # Set initial LR

    def step(self):
        if self.step_num < self.up_steps:
            pct = self.step_num / self.up_steps
            lr = self.initial_lr + pct * (self.max_lr - self.initial_lr)
        else:
            pct = (self.step_num - self.up_steps) / max(1, self.down_steps)
            lr = self.max_lr - pct * (self.max_lr - self.final_lr)

        self.optimizer.lr = lr
        self.step_num += 1

# --- Model Wrapper ---
class Model:
    def __init__(self):
        self.model = FashionConvNet()
    def __call__(self, x):
        return self.model(x)
    def parameters(self):
        return self.model.parameters()
    def train(self):         # <-- Add this
        self.model.train()
    def eval(self):          # <-- Add this
        self.model.eval()

def train_step():
    model.train()
    opt.zero_grad_c()
    x, y = random_batch(mm_train, BATCH_SIZE, INPUT_SIZE, NUM_TRAIN_SAMPLES)
    out = model(x)
    loss = out.cross_entropy(y)
    loss.backward()
    opt.step()
    return loss

def get_test_acc():
    model.eval()
    correct = 0
    total = 0
    for i in range(0, NUM_TEST_SAMPLES, TEST_BATCH_SIZE):
        actual_bs = min(TEST_BATCH_SIZE, NUM_TEST_SAMPLES - i)
        indices = list(range(i, i + actual_bs))
        bx, by = build_batch_from_mmap(mm_test, indices, INPUT_SIZE)
        x = Tensor(bx, requires_grad=False, shape=(actual_bs, 1, 28, 28), size=actual_bs * INPUT_SIZE)
        y = Tensor(by, requires_grad=False, shape=(actual_bs,), size=actual_bs)
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
TEST_BATCH_SIZE = 512
STEPS = 300

if __name__ == '__main__':
    train_bin = 'deepgrad/examples/datasets/fashion_mnist_train.bin'
    test_bin  = 'deepgrad/examples/datasets/fashion_mnist_test.bin'
    if not os.path.exists(train_bin) or not os.path.exists(test_bin):
        convert_csv_to_bin('deepgrad/examples/datasets/fashion_mnist_train.csv', train_bin)
        convert_csv_to_bin('deepgrad/examples/datasets/fashion_mnist_test.csv', test_bin)
    mm_train, NUM_TRAIN_SAMPLES, _ = load_bin_dataset(train_bin, 60000, INPUT_SIZE)
    mm_test, NUM_TEST_SAMPLES, _ = load_bin_dataset(test_bin, 10000, INPUT_SIZE)

    model = Model()
    opt = Adam(model.parameters(), lr=0.001, beta1=0.9, beta2=0.99)
    sched = OneCycleLR(
            optimizer=opt,
            max_lr=0.01,
            total_steps=STEPS
        )

    for step in trange(STEPS, desc="Training"):
        loss = train_step()
        sched.step()  # ← update learning rate
        if step % 10 == 9:
            acc = get_test_acc()
            print(f"step: {step+1:03d} | loss: {loss.data[0]:.4f} | lr: {opt.lr:.5f} | test acc: {acc:.2f}%")
        if step % 300 == 300:
            print(f"test acc: {acc:.2f}%")

    save_model(model, 'deepgrad/examples/fashion_model.pkl')
    print("Model saved to fashion_model.pkl")
