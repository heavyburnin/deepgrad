from deepgrad.backend import SimdTensorBackend
from ctypes import c_float, c_int

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

    def set_lr(self, new_lr):
        self.lr = new_lr

class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.99, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.m = [(c_float * p.size)() for p in parameters]
        self.v = [(c_float * p.size)() for p in parameters]

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            if param.grad is None or not param.requires_grad:
                continue
            SimdTensorBackend.adam_update_inplace(
                param.data,
                param.grad,
                self.m[i],
                self.v[i],
                param.size,
                c_float(self.lr),
                c_float(self.beta1),
                c_float(self.beta2),
                c_float(self.eps),
                c_int(self.t)
            )

    def zero_grad_c(self):
        for param in self.parameters:
            if param.requires_grad and param.grad is not None:
                SimdTensorBackend.zero_float_array(param.grad, param.size)

    def set_lr(self, new_lr):
        self.lr = new_lr
        
class AdamW:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.001):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = [(c_float * p.size)() for p in parameters]
        self.v = [(c_float * p.size)() for p in parameters]

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            if param.grad is None or not param.requires_grad:
                continue

            # AdamW = weight decay applied directly to param
            SimdTensorBackend.adamw_update_inplace(
                param.data,
                param.grad,
                self.m[i],
                self.v[i],
                param.size,
                c_float(self.lr),
                c_float(self.beta1),
                c_float(self.beta2),
                c_float(self.eps),
                c_int(self.t),
                c_float(self.weight_decay),
            )

    def zero_grad_c(self):
        for param in self.parameters:
            if param.requires_grad and param.grad is not None:
                SimdTensorBackend.zero_float_array(param.grad, param.size)

    def set_lr(self, new_lr):
        self.lr = new_lr
