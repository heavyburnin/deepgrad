import math
from deepgrad.tensor import Tensor
from deepgrad.batchnorm import BatchNorm2D

class MNISTNet:
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

        if self.training:
            x = x.dropout(0.1)

        # Conv block 2
        x = x.conv2d(self.w2, self.b2, stride=(1, 1), padding=(0, 0))
        x = self.bn2(x).relu().maxpool2d(kernel_size=2, stride=2)

        if self.training:
            x = x.dropout(0.15)

        # Flatten and FC
        x = x.flatten(start_dim=1)
        x = (x.matmul(self.w3) + self.b3).relu()

        if self.training:
            x = x.dropout(0.25)

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

class FashionNet:
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

        if self.training:
            x = x.dropout(0.25)

        # Conv block 2
        x = x.conv2d(self.w2, self.b2, stride=(1, 1), padding=(0, 0))
        x = self.bn2(x).relu().maxpool2d(kernel_size=2, stride=2)

        if self.training:
            x = x.dropout(0.3)

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

class ChessNet:
    def __init__(self):
        self.training = True

        # Chess board input: 8x8 board with 12 channels (6 piece types x 2 colors)
        # Conv1: 12 → 64, 3x3
        self.w1 = Tensor.randn((64, 12, 3, 3), std=math.sqrt(2 / (12 * 3 * 3)), requires_grad=True)
        self.b1 = Tensor.zeros((64,), requires_grad=True)
        self.bn1 = BatchNorm2D(64)

        # Conv2: 64 → 128, 3x3
        self.w2 = Tensor.randn((128, 64, 3, 3), std=math.sqrt(2 / (64 * 3 * 3)), requires_grad=True)
        self.b2 = Tensor.zeros((128,), requires_grad=True)
        self.bn2 = BatchNorm2D(128)

        # FC1: 128 * 2 * 2 → 512 (after 2x maxpool: 8x8 → 4x4 → 2x2)
        self.w3 = Tensor.randn((128 * 2 * 2, 512), std=math.sqrt(2 / (128 * 2 * 2)), requires_grad=True)
        self.b3 = Tensor.zeros((1, 512), requires_grad=True)

        # FC2: 512 → 3 (classification for win/loss/draw)
        self.w4 = Tensor.randn((512, 3), std=math.sqrt(2 / 512), requires_grad=True)
        self.b4 = Tensor.zeros((1, 3), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        # Input x: (batch_size, 12, 8, 8) for 12 piece channels, 8x8 board
        # Conv block 1
        x = x.conv2d(self.w1, self.b1, stride=(1, 1), padding=(1, 1))  # Output: (batch_size, 64, 8, 8)
        x = self.bn1(x).relu().maxpool2d(kernel_size=2, stride=2)  # Output: (batch_size, 64, 4, 4)

        # Conv block 2
        x = x.conv2d(self.w2, self.b2, stride=(1, 1), padding=(1, 1))  # Output: (batch_size, 128, 4, 4)
        x = self.bn2(x).relu().maxpool2d(kernel_size=2, stride=2)  # Output: (batch_size, 128, 2, 2)

        # Flatten and FC
        x = x.flatten(start_dim=1)  # Shape: (batch_size, 128 * 2 * 2)
        x = (x.matmul(self.w3) + self.b3).relu()

        # Output: 3 values for classification (win/loss/draw)
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
class FashionConvNetasdf:
    def __init__(self):
        self.training = True

        # Conv1: 1 -> 32 channels, 5x5 kernel, padding=2 (same padding)
        self.w1 = Tensor.randn((32, 1, 5, 5), std=math.sqrt(2 / (1 * 5 * 5)), requires_grad=True)
        self.b1 = Tensor.zeros((32,), requires_grad=True)
        self.bn1 = BatchNorm2D(32)

        # Conv2: 32 -> 48 channels, 3x3 kernel, padding=1 (same padding)
        self.w2 = Tensor.randn((48, 32, 3, 3), std=math.sqrt(2 / (32 * 3 * 3)), requires_grad=True)
        self.b2 = Tensor.zeros((48,), requires_grad=True)
        self.bn2 = BatchNorm2D(48)

        # Conv3: 48 -> 64 channels, 3x3 kernel, padding=1 (same padding)
        self.w3 = Tensor.randn((64, 48, 3, 3), std=math.sqrt(2 / (48 * 3 * 3)), requires_grad=True)
        self.b3 = Tensor.zeros((64,), requires_grad=True)
        self.bn3 = BatchNorm2D(64)

        # After two 2x2 maxpool layers, input spatial size goes 28 -> 14 -> 7
        # Flatten size = 7 * 7 * 64 = 3136
        self.w4 = Tensor.randn((3136, 256), std=math.sqrt(2 / 3136), requires_grad=True)
        self.b4 = Tensor.zeros((256,), requires_grad=True)

        self.w5 = Tensor.randn((256, 10), std=math.sqrt(2 / 256), requires_grad=True)
        self.b5 = Tensor.zeros((10,), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        x = x.reshape((x.shape[0], 1, 28, 28))  # NCHW

        # Conv block 1
        x = x.conv2d(self.w1, self.b1, stride=(1, 1), padding=(2, 2))
        x = self.bn1(x).relu()
        if self.training:
            x = x.dropout(0.5)
        x = x.maxpool2d(kernel_size=2, stride=2)  # 28 -> 14

        # Conv block 2
        x = x.conv2d(self.w2, self.b2, stride=(1, 1), padding=(1, 1))
        x = self.bn2(x).relu()
        if self.training:
            x = x.dropout(0.2)
        x = x.maxpool2d(kernel_size=2, stride=2)  # 14 -> 7

        # Conv block 3 (no pooling here)
        x = x.conv2d(self.w3, self.b3, stride=(1, 1), padding=(1, 1))
        x = self.bn3(x).relu()
        if self.training:
            x = x.dropout(0.25)

        # Flatten
        x = x.flatten(start_dim=1)
        # FC1
        x = (x.matmul(self.w4) + self.b4).relu()
        if self.training:
            x = x.dropout(0.5)
        # FC2 output logits
        return x.matmul(self.w5) + self.b5

    def parameters(self):
        return [
            self.w1, self.b1,
            self.w2, self.b2,
            self.w3, self.b3,
            self.w4, self.b4,
            self.w5, self.b5,
            *self.bn1.parameters(),
            *self.bn2.parameters(),
            *self.bn3.parameters(),
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
