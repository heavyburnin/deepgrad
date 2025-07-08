import random
import os
import dill
import mmap
from tqdm import tqdm
from array import array
from ctypes import c_float, POINTER, cast, memmove, addressof, c_int
from deepgrad.tensor import Tensor
from deepgrad.model import ConvNetBatchedNorm
from deepgrad.optimizer import Adam

def convert_csv_to_bin(csv_path, bin_path):
    FASHION_MNIST_MEAN = 0.2860406
    FASHION_MNIST_STD = 0.3530242

    with open(csv_path, 'r') as f_csv, open(bin_path, 'wb') as f_bin:
        next(f_csv)  # Skip header
        for line in f_csv:
            values = line.strip().split(',')
            label_idx = int(values[0])  # integer class label 0–9

            # Write label as 4-byte int
            f_bin.write(c_int(label_idx))

            # Normalize image and write as float32
            image = array('f', ((float(px) / 255.0 - FASHION_MNIST_MEAN) / FASHION_MNIST_STD for px in values[1:]))
            f_bin.write(image.tobytes())

def load_bin_dataset(bin_path, num_samples, input_size):
    sample_size = input_size + 1  # 1 label (int stored as float32)
    with open(bin_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        actual_bytes = mm.size()
        expected_bytes = num_samples * sample_size * 4

        if actual_bytes < expected_bytes:
            print(f"[WARN] File smaller than expected. Shrinking num_samples.")
            num_samples = actual_bytes // (sample_size * 4)

        return mm, num_samples, sample_size

def build_batch_from_mmap(mm, sample_indices, input_size):
    sample_bytes = 4 + input_size * 4  # 4 bytes for int32 label, 4*784 for image
    batch_size = len(sample_indices)

    total_input = batch_size * input_size
    x_array = (c_float * total_input)()
    y_array = (c_int * batch_size)()

    for i, sample_idx in enumerate(sample_indices):
        offset = sample_idx * sample_bytes

        # Copy label (int32)
        label_ptr = cast(addressof(y_array) + i * 4, POINTER(c_int))
        memmove(label_ptr, mm[offset : offset + 4], 4)

        # Copy input (float32 × 784)
        x_ptr = cast(addressof(x_array) + i * input_size * 4, POINTER(c_float))
        memmove(x_ptr, mm[offset + 4 : offset + sample_bytes], input_size * 4)

    return x_array, y_array

def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        dill.dump(model, f)

def accuracy(pred, target):
    logits = pred.data
    targets = target.data
    num_classes = 10
    batch_size = len(targets)

    correct = 0
    for j in range(batch_size):
        pred_index = max(range(num_classes), key=lambda i: logits[j * num_classes + i])
        true_index = targets[j]
        if pred_index == true_index:
            correct += 1

    return correct / batch_size

def evaluate(model, test_path='deepgrad/examples/datasets/fashion_mnist_test.bin'):
    model.eval()
    input_size = 784
    batch_size = 512
    num_samples = 10000

    mm, num_samples, _ = load_bin_dataset(test_path, num_samples, input_size)

    total_loss = 0.0
    correct = 0.0
    total = 0

    for i in range(0, num_samples, batch_size):
        actual_batch_size = min(batch_size, num_samples - i)
        batch_indices = list(range(i, i + actual_batch_size))

        batch_x, batch_y = build_batch_from_mmap(mm, batch_indices, input_size)

        x_size = actual_batch_size * input_size
        y_size = actual_batch_size

        x = Tensor(batch_x, requires_grad=True, shape=(actual_batch_size, 1, 28, 28), size=x_size)
        y = Tensor(batch_y, requires_grad=False, shape=(actual_batch_size,), size=y_size)

        pred = model(x)
        loss = pred.cross_entropy(y)

        total_loss += loss.data[0]
        correct += accuracy(pred.detach(), y) * actual_batch_size
        total += actual_batch_size

    avg_loss = total_loss / total
    acc = (correct / total) * 100
    print(f"[Eval] Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

    return avg_loss, acc

def train():
    input_size = 784
    batch_size = 256
    num_epochs = 100

    model = ConvNetBatchedNorm()
    model.train()
    adam = Adam(model.parameters(), lr=0.001, beta1=0.7, beta2=0.9, eps=1e-8)

    mm, num_samples, _ = load_bin_dataset('deepgrad/examples/datasets/fashion_mnist_train.bin', 60000, input_size)

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0.0
        total = 0

        indices = list(range(num_samples))
        random.shuffle(indices)

        progress = tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch+1}/{num_epochs}", dynamic_ncols=False)

        for i in progress:
            actual_batch_size = min(batch_size, num_samples - i)
            batch_indices = indices[i:i + actual_batch_size]

            batch_x, batch_y = build_batch_from_mmap(mm, batch_indices, input_size)

            x = Tensor(batch_x, requires_grad=True, shape=(actual_batch_size, 1, 28, 28), size=actual_batch_size * input_size)
            y = Tensor(batch_y, requires_grad=False, shape=(actual_batch_size,), size=actual_batch_size)

            pred = model(x)
            loss = pred.cross_entropy(y)
            loss.backward()
            adam.step()
            adam.zero_grad_c()

            total_loss += loss.data[0] * actual_batch_size
            correct += accuracy(pred.detach(), y) * actual_batch_size
            total += actual_batch_size

            if i % (batch_size * 25) == 0:
                progress.set_postfix({
                    "loss": total_loss / (total or 1),
                    "acc": f"{(correct / total) * 100:.2f}%"
                })

        # print(f"Epoch {epoch+1}: Loss={total_loss / total:.4f}, Accuracy={(correct / total) * 100:.2f}%")

        if epoch == num_epochs - 1:
            evaluate(model)

    save_model(model, 'deepgrad/examples/fashion_model.pkl')
    print("Model saved to fashion_model.pkl")

if __name__ == '__main__':
    if not os.path.exists('deepgrad/examples/datasets/fashion_mnist_train.bin') or not os.path.exists('deepgrad/examples/datasets/fashion_mnist_test.bin'):
        convert_csv_to_bin('deepgrad/examples/datasets/fashion_mnist_train.csv', 'deepgrad/examples/datasets/fashion_mnist_train.bin')
        convert_csv_to_bin('deepgrad/examples/datasets/fashion_mnist_test.csv', 'deepgrad/examples/datasets/fashion_mnist_test.bin')
    else:
        print("Binary file already exists. Skipping conversion.")

    train()