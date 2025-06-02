import random
import math
from tensor import Tensor

class Linear:
    def __init__(self, in_dim, out_dim):
        limit = math.sqrt(6 / (in_dim + out_dim))  # Xavier Glorot Uniform
        self.weight = Tensor([random.uniform(-limit, limit) for _ in range(in_dim * out_dim)], requires_grad=True, shape=(in_dim, out_dim))
        self.bias = Tensor([0.0 for _ in range(out_dim)], requires_grad=True, shape=(1, out_dim))
        self.in_dim = in_dim
        self.out_dim = out_dim

    def __call__(self, x):
        out = x.matmul_tra(self.weight)
        return out + self.bias  # bias broadcast across batch

    def parameters(self):
        return [self.weight, self.bias]

class ReLU:
    def __call__(self, x):
        return x.relu()

class MLP:
    def __init__(self, input_size, hidden1, hidden2, output_size):
        self.fc1 = Linear(input_size, hidden1)
        self.relu1 = ReLU()
        self.fc2 = Linear(hidden1, hidden2)
        self.relu2 = ReLU()
        self.fc3 = Linear(hidden2, output_size)

    def __call__(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters() + self.fc3.parameters()