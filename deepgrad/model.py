import math
import random
from deepgrad.tensor import Tensor
from deepgrad.batchnorm import BatchNorm2D
from ctypes import c_float

def make_random_array(size, scale):
    arr = (c_float * size)()
    for i in range(size):
        arr[i] = random.uniform(-scale, scale)
    return arr

class MNISTConvNet:
    def __init__(self):
        self.training = True

        # Conv1: 1 → 24, 5x5
        self.w1 = Tensor.randn((24, 1, 5, 5), std=math.sqrt(2 / (1 * 5 * 5)), requires_grad=True)
        self.b1 = Tensor.zeros((24,), requires_grad=True)
        self.bn1 = BatchNorm2D(24)

        # Conv2: 24 → 32, 3x3
        self.w2 = Tensor.randn((32, 24, 3, 3), std=math.sqrt(2 / (24 * 3 * 3)), requires_grad=True)
        self.b2 = Tensor.zeros((32,), requires_grad=True)
        self.bn2 = BatchNorm2D(32)

        # FC1: 800 → 256 (after 2x maxpool on 28x28)
        self.w3 = Tensor.randn((800, 256), std=math.sqrt(2 / 800), requires_grad=True)
        self.b3 = Tensor.zeros((1, 256), requires_grad=True)

        # FC2: 256 → 10
        self.w4 = Tensor.randn((256, 10), std=math.sqrt(2 / 256), requires_grad=True)
        self.b4 = Tensor.zeros((1, 10), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        x = x.reshape((x.shape[0], 1, 28, 28))  # NCHW

        # Conv block 1
        x = x.conv2d(self.w1, self.b1, stride=(1, 1), padding=(0, 0))
        x = self.bn1(x).relu().maxpool2d(kernel_size=2, stride=2)

        # Conv block 2
        x = x.conv2d(self.w2, self.b2, stride=(1, 1), padding=(0, 0))
        x = self.bn2(x).relu().maxpool2d(kernel_size=2, stride=2)

        # Flatten and FC
        x = x.flatten(start_dim=1)
        x = (x.matmul(self.w3) + self.b3).relu()

        if self.training:
            x = x.dropout(0.5)

        return x.matmul(self.w4) + self.b4

    def parameters(self):
        return [
            self.w1, self.b1,
            self.w2, self.b2,
            self.w3, self.b3,
            self.w4, self.b4,
            *self.bn1.parameters(),
            *self.bn2.parameters(),
        ]

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

        # Conv1: 1 → 32, kernel 5x5
        scale1 = math.sqrt(2.0 / (1 * 5 * 5))
        self.w1 = Tensor(make_random_array(32 * 1 * 5 * 5, scale1), requires_grad=True, shape=(32, 1, 5, 5))
        self.b1 = Tensor.zeros((32,), requires_grad=True)

        # Conv2: 32 → 32, kernel 5x5
        scale2 = math.sqrt(2.0 / (32 * 5 * 5))
        self.w2 = Tensor(make_random_array(32 * 32 * 5 * 5, scale2), requires_grad=True, shape=(32, 32, 5, 5))
        self.b2 = Tensor.zeros((32,), requires_grad=True)

        self.bn1 = BatchNorm2D(32)

        # Conv3: 32 → 64, kernel 3x3
        scale3 = math.sqrt(2.0 / (32 * 3 * 3))
        self.w3 = Tensor(make_random_array(64 * 32 * 3 * 3, scale3), requires_grad=True, shape=(64, 32, 3, 3))
        self.b3 = Tensor.zeros((64,), requires_grad=True)

        # Conv4: 64 → 64, kernel 3x3
        scale4 = math.sqrt(2.0 / (64 * 3 * 3))
        self.w4 = Tensor(make_random_array(64 * 64 * 3 * 3, scale4), requires_grad=True, shape=(64, 64, 3, 3))
        self.b4 = Tensor.zeros((64,), requires_grad=True)

        self.bn2 = BatchNorm2D(64)

        # FC1: 576 → 256
        scale5 = math.sqrt(2.0 / 576)
        self.w5 = Tensor(make_random_array(576 * 256, scale5), requires_grad=True, shape=(576, 256))
        self.b5 = Tensor.zeros((1, 256), requires_grad=True)

        # FC2: 256 → 10
        scale6 = math.sqrt(2.0 / 256)
        self.w6 = Tensor(make_random_array(256 * 10, scale6), requires_grad=True, shape=(256, 10))
        self.b6 = Tensor.zeros((1, 10), requires_grad=True)

    def __call__(self, x):
        x = x.reshape((x.shape[0], 1, 28, 28))

        x = x.conv2d(self.w1, self.b1, stride=(1, 1), padding=(0, 0)).relu()
        if self.training:
            x = x.dropout(0.2)

        x = x.conv2d(self.w2, self.b2, stride=(1, 1), padding=(0, 0)).relu()
        if self.training:
            x = x.dropout(0.2)

        x = self.bn1(x)
        x = x.maxpool2d(kernel_size=2, stride=2)

        x = x.conv2d(self.w3, self.b3, stride=(1, 1), padding=(0, 0)).relu()
        x = x.conv2d(self.w4, self.b4, stride=(1, 1), padding=(0, 0)).relu()

        x = self.bn2(x)
        x = x.maxpool2d(kernel_size=2, stride=2)

        x = x.flatten(start_dim=1)
        x = x.matmul(self.w5) + self.b5
        x = x.relu()
        if self.training:
            x = x.dropout(0.2)

        x = x.matmul(self.w6) + self.b6
        return x

    def parameters(self):
        return [
            self.w1, self.b1, self.w2, self.b2,
            self.w3, self.b3, self.w4, self.b4,
            self.w5, self.b5, self.w6, self.b6,
            *self.bn1.parameters(),
            *self.bn2.parameters()
        ]

    def train(self):
        self.training = True
        self.bn1.training = True
        self.bn2.training = True

    def eval(self):
        self.training = False
        self.bn1.training = False
        self.bn2.training = False
