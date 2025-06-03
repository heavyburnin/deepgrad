import random
import math
from tensor import Tensor

class Linear:
    def __init__(self, in_dim, out_dim):
        # Xavier Glorot Uniform initialization
        limit = math.sqrt(6 / (in_dim + out_dim))
        # Create weights in one operation
        self.weight = Tensor(
            [random.uniform(-limit, limit) for _ in range(in_dim * out_dim)], 
            requires_grad=True, 
            shape=(in_dim, out_dim)
        )
        # Initialize bias to zeros
        self.bias = Tensor([0.0] * out_dim, requires_grad=True, shape=(1, out_dim))
        self.in_dim = in_dim
        self.out_dim = out_dim

    def __call__(self, x):
        # Combine operations to reduce intermediate tensors
        return x.matmul(self.weight) + self.bias

    def parameters(self):
        return [self.weight, self.bias]

class ReLU:
    def __call__(self, x):
        return x.relu()

class MLP:
    def __init__(self, input_size, hidden1, hidden2, output_size):
        self.layers = [
            Linear(input_size, hidden1),
            ReLU(),
            Linear(hidden1, hidden2),
            ReLU(),
            Linear(hidden2, output_size)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params