from ctypes import c_float

from deepgrad.tensor import Tensor

import random

def zeros(shape, requires_grad=False):
    size = 1
    for dim in shape:
        size *= dim
    data = (c_float * size)(*([0.0] * size))
    return Tensor(data, requires_grad=requires_grad, shape=shape, size=size)

def ones(shape, requires_grad=False):
    size = 1
    for dim in shape:
        size *= dim
    data = (c_float * size)(*([1.0] * size))
    return Tensor(data, requires_grad=requires_grad, shape=shape, size=size)

def randn(shape, requires_grad=False, mean=0.0, std=1.0):
    size = 1
    for dim in shape:
        size *= dim
    # Box-Muller normal sampling
    def normal():
        u1, u2 = random.random(), random.random()
        z0 = (2 * (-1) ** int(u1 * 2)) * (abs(2 * u1 - 1) ** 0.5) * ((-2 * math.log(u2)) ** 0.5)
        return mean + std * z0
    import math
    data = (c_float * size)(*[random.gauss(mean, std) for _ in range(size)])
    return Tensor(data, requires_grad=requires_grad, shape=shape, size=size)

x = randn((1, 3, 28, 28), requires_grad=True)
w = randn((8, 3, 3, 3), requires_grad=True)
b = zeros((8,), requires_grad=True)

y = x.conv2d(w, b, stride=(1, 1), padding=(1, 1))
y.sum().backward()

print(x.grad[:10])  # gradients for first 10 input pixels
print(w.grad[:10])  # gradients for first 10 weights
print(b.grad[:])    # gradient for each bias
