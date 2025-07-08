import random
import math
from ctypes import c_float
from deepgrad.tensor import Tensor
from deepgrad.batchnorm import BatchNorm1D, BatchNorm2D

def make_random_array(size, scale):
    arr = (c_float * size)()
    for i in range(size):
        arr[i] = random.uniform(-scale, scale)
    return arr

def make_zero_array(size):
    return (c_float * size)()

class ConvNet:
    def __init__(self):
        self.training = True

        # Conv1: 1 → 24
        scale1 = math.sqrt(2.0 / (1 * 5 * 5))
        self.w1 = Tensor(make_random_array(24 * 1 * 5 * 5, scale1), requires_grad=True, shape=(24, 1, 5, 5), size=24 * 1 * 5 * 5)
        self.b1 = Tensor(make_zero_array(24), requires_grad=True, shape=(24,), size=24)
        self.bn1 = BatchNorm2D(24)

        # Conv2: 24 → 32
        scale2 = math.sqrt(2.0 / (24 * 3 * 3))
        self.w2 = Tensor(make_random_array(32 * 24 * 3 * 3, scale2), requires_grad=True, shape=(32, 24, 3, 3), size=32 * 24 * 3 * 3)
        self.b2 = Tensor(make_zero_array(32), requires_grad=True, shape=(32,), size=32)
        self.bn2 = BatchNorm2D(32)

        # FC1: 800 → 256
        scale3 = math.sqrt(2.0 / 800)
        self.w3 = Tensor(make_random_array(800 * 256, scale3), requires_grad=True, shape=(800, 256), size=800 * 256)
        self.b3 = Tensor(make_zero_array(256), requires_grad=True, shape=(1, 256), size=256)

        # FC2: 256 → 10
        scale4 = math.sqrt(2.0 / 256)
        self.w4 = Tensor(make_random_array(256 * 10, scale4), requires_grad=True, shape=(256, 10), size=256 * 10)
        self.b4 = Tensor(make_zero_array(10), requires_grad=True, shape=(1, 10), size=10)

        # Layer list
        self.layers = [
            lambda x: x.reshape((x.shape[0], 1, 28, 28)),

            lambda x: x.conv2d(self.w1, self.b1, stride=(1, 1), padding=(0, 0)),
            self.bn1, Tensor.relu,
            lambda x: x.maxpool2d(kernel_size=2, stride=2),

            lambda x: x.conv2d(self.w2, self.b2, stride=(1, 1), padding=(0, 0)),
            self.bn2, Tensor.relu,
            lambda x: x.maxpool2d(kernel_size=2, stride=2),

            lambda x: x.reshape((x.shape[0], -1)),
            lambda x: x.matmul(self.w3) + self.b3,
            Tensor.relu,
            lambda x: x.dropout(0.25) if self.training else x,

            lambda x: x.matmul(self.w4) + self.b4
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [self.w1, self.b1,
                self.w2, self.b2,
                self.w3, self.b3,
                self.w4, self.b4,
                *self.bn1.parameters(), *self.bn2.parameters()]

    def train(self):
        self.training = True
        self.bn1.training = True
        self.bn2.training = True

    def eval(self):
        self.training = False
        self.bn1.training = False
        self.bn2.training = False

class FashionConvNet:
    def __init__(self):
        self.training = True

        # Conv1: 1 → 24
        scale1 = math.sqrt(2.0 / (1 * 5 * 5))
        self.w1 = Tensor(make_random_array(24 * 1 * 5 * 5, scale1), requires_grad=True, shape=(24, 1, 5, 5), size=24 * 1 * 5 * 5)
        self.b1 = Tensor(make_zero_array(24), requires_grad=True, shape=(24,), size=24)
        self.bn1 = BatchNorm2D(24)

        # Conv2: 24 → 32
        scale2 = math.sqrt(2.0 / (24 * 3 * 3))
        self.w2 = Tensor(make_random_array(32 * 24 * 3 * 3, scale2), requires_grad=True, shape=(32, 24, 3, 3), size=32 * 24 * 3 * 3)
        self.b2 = Tensor(make_zero_array(32), requires_grad=True, shape=(32,), size=32)
        self.bn2 = BatchNorm2D(32)

        # FC1: 800 → 256
        scale3 = math.sqrt(2.0 / 800)
        self.w3 = Tensor(make_random_array(800 * 256, scale3), requires_grad=True, shape=(800, 256), size=800 * 256)
        self.b3 = Tensor(make_zero_array(256), requires_grad=True, shape=(1, 256), size=256)

        # FC2: 256 → 10
        scale4 = math.sqrt(2.0 / 256)
        self.w4 = Tensor(make_random_array(256 * 10, scale4), requires_grad=True, shape=(256, 10), size=256 * 10)
        self.b4 = Tensor(make_zero_array(10), requires_grad=True, shape=(1, 10), size=10)

        # Layer list
        self.layers = [
            lambda x: x.reshape((x.shape[0], 1, 28, 28)),

            lambda x: x.conv2d(self.w1, self.b1, stride=(1, 1), padding=(0, 0)),
            self.bn1, Tensor.relu,
            lambda x: x.maxpool2d(kernel_size=2, stride=2),

            lambda x: x.conv2d(self.w2, self.b2, stride=(1, 1), padding=(0, 0)),
            self.bn2, Tensor.relu,
            lambda x: x.maxpool2d(kernel_size=2, stride=2),

            lambda x: x.reshape((x.shape[0], -1)),
            lambda x: x.matmul(self.w3) + self.b3,
            Tensor.relu,
            lambda x: x.dropout(0.5) if self.training else x,

            lambda x: x.matmul(self.w4) + self.b4
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [self.w1, self.b1,
                self.w2, self.b2,
                self.w3, self.b3,
                self.w4, self.b4,
                *self.bn1.parameters(), *self.bn2.parameters()]

    def train(self):
        self.training = True
        self.bn1.training = True
        self.bn2.training = True

    def eval(self):
        self.training = False
        self.bn1.training = False
        self.bn2.training = False
