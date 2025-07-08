import random
import math
from ctypes import c_float
from deepgrad.tensor import Tensor
from deepgrad.batchnorm import BatchNorm2D

def make_random_array(size, scale):
    arr = (c_float * size)()
    for i in range(size):
        arr[i] = random.uniform(-scale, scale)
    return arr

def make_zero_array(size):
    return (c_float * size)()

class MNISTConvNet:
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

    def __call__(self, x):
        # Input reshape
        x = x.reshape((x.shape[0], 1, 28, 28))

        # Conv block 1
        x = x.conv2d(self.w1, self.b1, stride=(1, 1), padding=(0, 0))
        x = self.bn1(x)
        x = x.relu()
        x = x.maxpool2d(kernel_size=2, stride=2)

        # Conv block 2
        x = x.conv2d(self.w2, self.b2, stride=(1, 1), padding=(0, 0))
        x = self.bn2(x)
        x = x.relu()
        x = x.maxpool2d(kernel_size=2, stride=2)

        # Flatten
        x = x.reshape((x.shape[0], -1))

        # Fully connected block
        x = x.matmul(self.w3) + self.b3
        x = x.relu()
        if self.training:
            x = x.dropout(0.5)

        x = x.matmul(self.w4) + self.b4
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

class Test:
    def __init__(self):
        self.training = True

        # --- Conv Block 1: 1 → 32 ---
        scale1 = math.sqrt(2.0 / (1 * 5 * 5))
        self.w1 = Tensor(make_random_array(32 * 1 * 5 * 5, scale1), requires_grad=True, shape=(32, 1, 5, 5), size=32 * 1 * 5 * 5)
        self.b1 = Tensor(make_zero_array(32), requires_grad=True, shape=(32,), size=32)
        self.bn1 = BatchNorm2D(32)

        # --- Conv Block 2: 32 → 64 ---
        scale2 = math.sqrt(2.0 / (32 * 3 * 3))
        self.w2 = Tensor(make_random_array(64 * 32 * 3 * 3, scale2), requires_grad=True, shape=(64, 32, 3, 3), size=64 * 32 * 3 * 3)
        self.b2 = Tensor(make_zero_array(64), requires_grad=True, shape=(64,), size=64)
        self.bn2 = BatchNorm2D(64)

        # --- Conv Block 3: 64 → 128 ---
        scale3 = math.sqrt(2.0 / (64 * 3 * 3))
        self.w3 = Tensor(make_random_array(128 * 64 * 3 * 3, scale3), requires_grad=True, shape=(128, 64, 3, 3), size=128 * 64 * 3 * 3)
        self.b3 = Tensor(make_zero_array(128), requires_grad=True, shape=(128,), size=128)
        self.bn3 = BatchNorm2D(128)

        # --- Fully Connected Layer 1: (128 × 3 × 3) → 512 ---
        scale_fc1 = math.sqrt(2.0 / (128 * 3 * 3))
        self.w_fc1 = Tensor(make_random_array(1152 * 512, scale_fc1), requires_grad=True, shape=(1152, 512), size=1152 * 512)
        self.b_fc1 = Tensor(make_zero_array(512), requires_grad=True, shape=(1, 512), size=512)

        # --- Fully Connected Layer 2: 512 → 10 ---
        scale_fc2 = math.sqrt(2.0 / 512)
        self.w_fc2 = Tensor(make_random_array(512 * 10, scale_fc2), requires_grad=True, shape=(512, 10), size=512 * 10)
        self.b_fc2 = Tensor(make_zero_array(10), requires_grad=True, shape=(1, 10), size=10)
    
    def __call__(self, x):
        x = x.reshape((x.shape[0], 1, 28, 28))

        x = x.conv2d(self.w1, self.b1, stride=(1,1), padding=(2,2))
        x = self.bn1(x).relu()
        x = x.maxpool2d(kernel_size=2, stride=2)
        if self.training: x = x.dropout(0.1)

        x = x.conv2d(self.w2, self.b2, stride=(1,1), padding=(1,1))
        x = self.bn2(x).relu()
        x = x.maxpool2d(kernel_size=2, stride=2)
        if self.training: x = x.dropout(0.25)

        x = x.conv2d(self.w3, self.b3, stride=(1,1), padding=(1,1))
        x = self.bn3(x).relu()
        x = x.maxpool2d(kernel_size=2, stride=2)
        if self.training: x = x.dropout(0.25)

        x = x.reshape((x.shape[0], -1))
        x = x.matmul(self.w_fc1) + self.b_fc1
        x = x.relu()
        if self.training: x = x.dropout(0.5)

        x = x.matmul(self.w_fc2) + self.b_fc2
        return x


    def parameters(self):
        return [
            self.w1, self.b1, self.w2, self.b2, self.w3, self.b3,
            self.w_fc1, self.b_fc1,
            self.w_fc2, self.b_fc2,
            *self.bn1.parameters(), *self.bn2.parameters(), *self.bn3.parameters()
        ]

    def train(self):
        self.training = True
        self.bn1.training = True
        self.bn2.training = True
        self.bn3.training = True

    def eval(self):
        self.training = False
        self.bn1.training = False
        self.bn2.training = False
        self.bn3.training = False
