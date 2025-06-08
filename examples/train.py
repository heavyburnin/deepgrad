import struct
import random
import os
import dill
from tensor import Tensor
from model import MLP
from optimizer import SGD
from tqdm import tqdm
import cProfile
import pstats
import array
import mmap

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
    expected_floats = num_samples * sample_size
    expected_bytes = expected_floats * 4  # float32 = 4 bytes

    with open(bin_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)  # map full file
        actual_bytes = os.fstat(f.fileno()).st_size

        if actual_bytes < expected_bytes:
            print(f"[WARN] File is smaller than expected. Shrinking num_samples.")
            num_samples = actual_bytes // (sample_size * 4)
            expected_floats = num_samples * sample_size
            expected_bytes = expected_floats * 4

        data = array.array('f')
        data.frombytes(mm[:expected_bytes])
        mm.close()

    return data, num_samples
        
def save_model(model, filepath):
    """Save model using dill"""
    with open(filepath, 'wb') as f:
        dill.dump(model, f)

def load_model(filepath):
    """Load model using dill"""
    with open(filepath, 'rb') as f:
        return dill.load(f)

def accuracy(pred, target):
    logits = list(pred.data)
    pred_class = [max(range(10), key=lambda i: logits[i + j * 10]) for j in range(len(logits) // 10)]
    true_class = [max(range(10), key=lambda i: target.data[i + j * 10]) for j in range(len(target.data) // 10)]
    return sum([int(p == t) for p, t in zip(pred_class, true_class)]) / len(pred_class)

def evaluate(model, test_path='mnist_test.bin'):
    input_size = 784
    output_size = 10
    sample_size = input_size + output_size
    num_samples = 10000
    batch_size = 32

    test_data, num_samples = load_bin_dataset(test_path, num_samples, sample_size)

    total_loss = 0.0
    correct = 0.0
    total = 0.0

    for i in range(0, num_samples, batch_size):
        actual_batch_size = min(batch_size, num_samples - i)

        batch_x = array.array('f', [0.0] * (actual_batch_size * input_size))
        batch_y = array.array('f', [0.0] * (actual_batch_size * output_size))

        for j in range(actual_batch_size):
            idx = i + j
            offset = idx * sample_size
            sample = test_data[offset : offset + sample_size]

            batch_x[j*input_size : (j+1)*input_size] = sample[:input_size]
            batch_y[j*output_size : (j+1)*output_size] = sample[input_size:]

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

def trace_graph(tensor, depth=0, seen=None):
    if seen is None:
        seen = set()
    if id(tensor) in seen:
        return
    seen.add(id(tensor))
    print("  " * depth + f"Tensor: id={id(tensor)} data={list(tensor.data)} requires_grad={tensor.requires_grad}")
    for p in tensor._prev:
        trace_graph(p, depth + 1, seen)

def train():
    input_size = 784
    output_size = 10
    hidden1 = 128
    hidden2 = 64
    num_samples = 60000
    batch_size = 32

    model = MLP(input_size, hidden1, hidden2, output_size)
    optimizer = SGD(model.parameters(), lr=0.01)

    #train_data = load_bin_dataset('mnist_train.bin', num_samples, input_size)
    train_data, num_samples = load_bin_dataset('mnist_train.bin', num_samples, input_size + 10)

    sample_size = input_size + 10

    for epoch in range(10):
        total_loss = 0.0
        correct = 0.0
        total = 0.0

        indices = list(range(num_samples))
        random.shuffle(indices)

        progress = tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch+1}/10", dynamic_ncols=False)

        for i in progress:
            batch_indices = indices[i:i+batch_size]
            actual_batch_size = len(batch_indices)

            batch_x = array.array('f', [0.0] * (actual_batch_size * input_size))
            batch_y = array.array('f', [0.0] * (actual_batch_size * output_size))

            for j, idx in enumerate(batch_indices):
                offset = idx * sample_size
                sample = train_data[offset : offset + sample_size]

                batch_x[j*input_size : (j+1)*input_size] = sample[:input_size]
                batch_y[j*output_size : (j+1)*output_size] = sample[input_size:]


            x = Tensor(batch_x, requires_grad=True, shape=(actual_batch_size, input_size))
            y = Tensor(batch_y, shape=(actual_batch_size, output_size))


            pred = model(x)
            loss = pred.cross_entropy(y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Break graph references
            loss._prev.clear()
            pred._prev.clear()
            x._prev.clear()

            total_loss += loss.data[0]
            correct += accuracy(pred, y) * actual_batch_size
            total += actual_batch_size

            if i % 25 == 0:
                progress.set_postfix({
                    "loss": total_loss / (total or 1),
                    "acc": f"{(correct / total) * 100:.2f}%"
                })

        evaluate(model)

    save_model(model, 'model.pkl')
    print("Model saved to model.pkl")

#if __name__ == '__main__':
#    if not os.path.exists('mnist_train.bin') or not os.path.exists('mnist_test.bin'):
#        convert_csv_to_bin('mnist_train.csv', 'mnist_train.bin')
#        convert_csv_to_bin('mnist_test.csv', 'mnist_test.bin')
#    else:
#        print("Binary file already exists. Skipping conversion.")

#    train()

if __name__ == '__main__':
    with cProfile.Profile() as pr:
        if not os.path.exists('mnist_train.bin') or not os.path.exists('mnist_test.bin'):
            convert_csv_to_bin('mnist_train.csv', 'mnist_train.bin')
            convert_csv_to_bin('mnist_test.csv', 'mnist_test.bin')
        else:
            print("Binary file already exists. Skipping conversion.")
        train()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)