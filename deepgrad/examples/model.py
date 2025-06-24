import random
import math
from ctypes import c_float
from deepgrad.tensor import Tensor

def make_random_array(size, scale):
    arr = (c_float * size)()
    for i in range(size):
        arr[i] = random.uniform(-scale, scale)
    return arr

def make_zero_array(size):
    return (c_float * size)()

class ConvNet:
    def __init__(self):
        scale1 = math.sqrt(2.0 / (1 * 3 * 3))
        self.w1 = Tensor(make_random_array(1 * 8 * 3 * 3, scale1), requires_grad=True, shape=(8, 1, 3, 3), size=8 * 1 * 3 * 3)
        self.b1 = Tensor(make_zero_array(8), requires_grad=True, shape=(8,), size=8)

        scale2 = math.sqrt(2.0 / (8 * 3 * 3))
        self.w2 = Tensor(make_random_array(8 * 16 * 3 * 3, scale2), requires_grad=True, shape=(16, 8, 3, 3), size=16 * 8 * 3 * 3)
        self.b2 = Tensor(make_zero_array(16), requires_grad=True, shape=(16,), size=16)

        scale3 = math.sqrt(2.0 / (16 * 7 * 7))
        self.w3 = Tensor(make_random_array(16 * 7 * 7 * 10, scale3), requires_grad=True, shape=(16 * 7 * 7, 10), size=16 * 7 * 7 * 10)
        self.b3 = Tensor(make_zero_array(10), requires_grad=True, shape=(1, 10), size=10)

    def __call__(self, x):
        x = x.reshape((x.shape[0], 1, 28, 28))

        x = x.conv2d(self.w1, self.b1, stride=(1, 1), padding=(1, 1)).relu()
        x = x.maxpool2d(kernel_size=2, stride=2)

        x = x.conv2d(self.w2, self.b2, stride=(1, 1), padding=(1, 1)).relu()
        x = x.maxpool2d(kernel_size=2, stride=2)
        

        x = x.reshape((x.shape[0], -1))
        x = x.matmul(self.w3) + self.b3
        return x

    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

class MLP:
    def __init__(self, input_size, hidden1, hidden2, output_size):
        xavier_1 = math.sqrt(6.0 / (input_size + hidden1))
        xavier_2 = math.sqrt(6.0 / (hidden1 + hidden2))
        xavier_3 = math.sqrt(6.0 / (hidden2 + output_size))

        size_w1 = input_size * hidden1
        size_b1 = hidden1
        size_w2 = hidden1 * hidden2
        size_b2 = hidden2
        size_w3 = hidden2 * output_size
        size_b3 = output_size

        self.w1 = Tensor(make_random_array(size_w1, xavier_1), requires_grad=True, shape=(input_size, hidden1), size=size_w1)

        self.b1 = Tensor(make_zero_array(size_b1), requires_grad=True, shape=(1, hidden1), size=size_b1)

        self.w2 = Tensor(make_random_array(size_w2, xavier_2), requires_grad=True, shape=(hidden1, hidden2), size=size_w2)

        self.b2 = Tensor(make_zero_array(size_b2), requires_grad=True, shape=(1, hidden2), size=size_b2)

        self.w3 = Tensor(make_random_array(size_w3, xavier_3), requires_grad=True, shape=(hidden2, output_size), size=size_w3)

        self.b3 = Tensor(make_zero_array(size_b3), requires_grad=True, shape=(1, output_size), size=size_b3)

    def __call__(self, x):
        x = (x.matmul(self.w1) + self.b1).relu()
        x = (x.matmul(self.w2) + self.b2).relu()
        x = x.matmul(self.w3) + self.b3
        return x

    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]