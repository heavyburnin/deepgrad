import random
import math
from array import array
from tensor import Tensor

class MLP:
    def __init__(self, input_size, hidden1, hidden2, output_size):
        # Xavier/Glorot initialization for better gradient flow
        xavier_1 = math.sqrt(6.0 / (input_size + hidden1))
        xavier_2 = math.sqrt(6.0 / (hidden1 + hidden2))
        xavier_3 = math.sqrt(6.0 / (hidden2 + output_size))
        
        # Initialize with float32 arrays
        self.w1 = Tensor(array('f', (random.uniform(-xavier_1, xavier_1) for _ in range(input_size * hidden1))),
                         requires_grad=True, shape=(input_size, hidden1))
        self.b1 = Tensor(array('f', (0.0 for _ in range(hidden1))),
                         requires_grad=True, shape=(1, hidden1))
        
        self.w2 = Tensor(array('f', (random.uniform(-xavier_2, xavier_2) for _ in range(hidden1 * hidden2))),
                         requires_grad=True, shape=(hidden1, hidden2))
        self.b2 = Tensor(array('f', (0.0 for _ in range(hidden2))),
                         requires_grad=True, shape=(1, hidden2))
        
        self.w3 = Tensor(array('f', (random.uniform(-xavier_3, xavier_3) for _ in range(hidden2 * output_size))),
                         requires_grad=True, shape=(hidden2, output_size))
        self.b3 = Tensor(array('f', (0.0 for _ in range(output_size))),
                         requires_grad=True, shape=(1, output_size))

    def __call__(self, x):
        # Forward pass through the network
        x = (x.matmul(self.w1) + self.b1).relu()
        x = (x.matmul(self.w2) + self.b2).relu()
        x = x.matmul(self.w3) + self.b3
        return x

    def parameters(self):
        # Return all trainable parameters
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
