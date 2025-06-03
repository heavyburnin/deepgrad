import dill
from tensor import Tensor

# Load model
with open('model.pkl', 'rb') as f:
    model = dill.load(f)

# Example inference
# Let's say you want to classify the first test sample
from train import load_bin_dataset

data = load_bin_dataset('mnist_test.bin', 1, 784 + 10)
x_data = data[0][:784]
x = Tensor(x_data, requires_grad=False, shape=(1, 784))
pred = model(x)

# Interpret prediction
logits = list(pred.data)
predicted_class = max(range(10), key=lambda i: logits[i])
print(f"Predicted digit: {predicted_class}")

