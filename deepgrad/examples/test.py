import numpy as np
from ctypes import c_float

from deepgrad.tensor import Tensor

# Create a ctypes float array with 3 elements
data = (c_float * 3)(1.0, 2.0, 3.0)

# Wrap it in a Tensor
x = Tensor(data, requires_grad=True, shape=(3,), size=3)

# Apply exponential
y = x.exp()

print("Input x       :", list(x.data))
print("exp(x) output :", list(y.data))

# Backprop on scalar sum
y.sum().backward()

print("x.grad        :", list(x.grad))
print("Expected grad :", list(np.exp([1.0, 2.0, 3.0])))