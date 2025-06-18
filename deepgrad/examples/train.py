import random
import os
import dill
from deepgrad.examples.model import MLP
from deepgrad.examples.optimizer import SGD
from tqdm import tqdm
import array
import mmap
import cProfile
import pstats
from deepgrad.tensor import Tensor

def convert_csv_to_bin(csv_path, bin_path):
    with open(csv_path, 'r') as f_csv, open(bin_path, 'wb') as f_bin:
        next(f_csv)  # skip header
        for line in f_csv:
            values = line.strip().split(',')
            label_idx = int(values[0])

            one_hot = array.array('f', (0.0 for _ in range(10)))
            one_hot[label_idx] = 1.0

            image = array.array('f', (float(px) / 255.0 for px in values[1:]))

            sample = image + one_hot  # array supports concatenation

            f_bin.write(sample.tobytes())

def load_bin_dataset(bin_path, num_samples, sample_size):
    with open(bin_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        actual_bytes = mm.size()
        expected_bytes = num_samples * sample_size * 4

        if actual_bytes < expected_bytes:
            print(f"[WARN] File smaller than expected. Shrinking num_samples.")
            num_samples = actual_bytes // (sample_size * 4)

        return mm, num_samples, sample_size

def build_batch_from_mmap(mm, sample_indices, input_size, output_size):
    sample_size = input_size + output_size
    sample_bytes = sample_size * 4
    x_array = array.array('f')
    y_array = array.array('f')

    for sample_idx in sample_indices:
        offset = sample_idx * sample_bytes
        raw = mm[offset : offset + sample_bytes]

        sample = array.array('f')
        sample.frombytes(raw)

        x_array.extend(sample[:input_size])
        y_array.extend(sample[input_size:])

    return x_array, y_array

def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        dill.dump(model, f)

def load_model(filepath):
    with open(filepath, 'rb') as f:
        return dill.load(f)

def accuracy(pred, target):
    logits = pred.data
    targets = target.data
    num_classes = 10
    batch_size = len(logits) // num_classes

    correct = 0
    for j in range(batch_size):
        pred_index = max(range(num_classes), key=lambda i: logits[j * num_classes + i])
        true_index = max(range(num_classes), key=lambda i: targets[j * num_classes + i])
        if pred_index == true_index:
            correct += 1
    
    return correct / batch_size

def evaluate(model, test_path='deepgrad/examples/mnist_test.bin'):
    input_size = 784
    output_size = 10
    sample_size = input_size + output_size
    batch_size = 32
    num_samples = 10000

    mm, num_samples, _ = load_bin_dataset(test_path, num_samples, sample_size)

    total_loss = 0.0
    correct = 0.0
    total = 0

    for i in range(0, num_samples, batch_size):
        actual_batch_size = min(batch_size, num_samples - i)
        batch_indices = list(range(i, i + actual_batch_size))

        batch_x, batch_y = build_batch_from_mmap(mm, batch_indices, input_size, output_size)
        x = Tensor(batch_x, requires_grad=True, shape=(actual_batch_size, input_size))
        y = Tensor(batch_y, shape=(actual_batch_size, output_size))

        pred = model(x)
        loss = pred.cross_entropy(y)

        total_loss += loss.data[0]
        correct += accuracy(pred, y) * actual_batch_size
        total += actual_batch_size

    avg_loss = total_loss / total
    acc = (correct / total) * 100
    print(f"[Eval] Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

    return avg_loss, acc

def train():
    input_size = 784
    output_size = 10
    hidden1 = 128
    hidden2 = 64
    batch_size = 32
    num_epochs = 20
    sample_size = input_size + output_size

    model = MLP(input_size, hidden1, hidden2, output_size)
    optimizer = SGD(model.parameters(), lr=0.01)

    mm, num_samples, _ = load_bin_dataset('deepgrad/examples/mnist_train.bin', 60000, sample_size)

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0.0
        total = 0

        indices = list(range(num_samples))
        random.shuffle(indices)

        progress = tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch+1}/{num_epochs}", dynamic_ncols=False)

        for i in progress:
            batch_indices = indices[i:i+batch_size]
            actual_batch_size = len(batch_indices)

            batch_x, batch_y = build_batch_from_mmap(mm, batch_indices, input_size, output_size)
            x = Tensor(batch_x, requires_grad=True, shape=(actual_batch_size, input_size))
            y = Tensor(batch_y, shape=(actual_batch_size, output_size))

            pred = model(x)
            loss = pred.cross_entropy(y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Clear graph refs to avoid memory leak
            loss._prev.clear()
            pred._prev.clear()

            total_loss += loss.data[0]
            correct += accuracy(pred, y) * actual_batch_size
            total += actual_batch_size

            if i % (batch_size * 25) == 0:
                progress.set_postfix({
                    "loss": total_loss / (total or 1),
                    "acc": f"{(correct / total) * 100:.2f}%"
                })

        if epoch == num_epochs - 1:
            evaluate(model)

    save_model(model, 'deepgrad/examples/model.pkl')
    print("Model saved to model.pkl")

if __name__ == '__main__':
    if not os.path.exists('deepgrad/examples/mnist_train.bin') or not os.path.exists('deepgrad/examples/mnist_test.bin'):
        convert_csv_to_bin('deepgrad/examples/mnist_train.csv', 'deepgrad/examples/mnist_train.bin')
        convert_csv_to_bin('deepgrad/examples/mnist_test.csv', 'deepgrad/examples/mnist_test.bin')
    else:
        print("Binary file already exists. Skipping conversion.")

    with cProfile.Profile() as pr:
        train()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)