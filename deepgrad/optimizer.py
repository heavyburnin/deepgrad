from deepgrad.backend import SimdTensorBackend
from ctypes import c_float

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                SimdTensorBackend.sgd_update_inplace(
                    param.data,
                    param.grad,
                    param.size,
                    c_float(self.lr)
                )

    def zero_grad_c(self):
        for param in self.parameters:
            if param.requires_grad and param.grad is not None:
                SimdTensorBackend.zero_float_array(
                    param.grad,
                    param.size
                )