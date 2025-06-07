import struct
import random
import os
import dill
from tensor import Tensor
from model import MLP
from optimizer import SGD
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
##remove below
import cProfile
import pstats

# CSV to .bin conversion (run once)
def convert_csv_to_bin(csv_path, bin_path):
    with open(csv_path, 'r') as f_csv, open(bin_path, 'wb') as f_bin:
        next(f_csv)  # Skip header
        for line in f_csv:
            values = list(map(float, line.strip().split(',')))
            label_idx = int(values[0])
            one_hot = [0.0] * 10
            one_hot[label_idx] = 1.0
            #image = values[1:]  # 784 values
            image = [px / 255.0 for px in values[1:]]
            sample = image + one_hot  # total: 784 + 10
            f_bin.write(struct.pack(f'{len(sample)}f', *sample))

def load_bin_dataset(bin_path, num_samples, sample_size):
    data = []
    with open(bin_path, 'rb') as f:
        for _ in range(num_samples):
            raw = f.read(sample_size * 4)
            sample = list(struct.unpack(f'{sample_size}f', raw))
            data.append(sample)
    return data

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
    num_samples = 10000  # MNIST test set
    batch_size = 32

    test_data = load_bin_dataset(test_path, num_samples, input_size + output_size)

    total_loss = 0.0
    correct = 0.0
    total = 0.0

    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i+batch_size]
        batch_x = [s[:input_size] for s in batch]
        batch_y = [s[input_size:] for s in batch]

        x = Tensor([val for row in batch_x for val in row], requires_grad=False, shape=(len(batch), input_size))
        y = Tensor([val for row in batch_y for val in row], shape=(len(batch), output_size))

        pred = model(x)
        loss = pred.cross_entropy(y)

        total_loss += loss.data[0]
        correct += accuracy(pred, y) * len(batch)
        total += len(batch)

    print(f"\nEvaluation - Loss: {total_loss / total:.4f}, Accuracy: {correct / total * 100:.2f}%")

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

    train_data = load_bin_dataset('mnist_train.bin', num_samples, input_size + 10)

    for epoch in range(10):
        total_loss = 0.0
        correct = 0.0
        total = 0.0
        random.shuffle(train_data)
        #progress = tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch+1}/10")
        progress = tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch+1}/10", dynamic_ncols=False)

        for i in progress:
            batch = train_data[i:i+batch_size]
            batch_x = [s[:input_size] for s in batch]
            batch_y = [s[input_size:] for s in batch]
            

            x = Tensor([val for row in batch_x for val in row], requires_grad=True, shape=(len(batch), input_size))
            y = Tensor([val for row in batch_y for val in row], shape=(len(batch), output_size))

            pred = model(x)
            loss = pred.cross_entropy(y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Break graph references to avoid memory buildup
            loss._prev.clear()
            pred._prev.clear()
            x._prev.clear()

            total_loss += loss.data[0]
            correct += accuracy(pred, y) * len(batch)
            total += len(batch)
            
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