import array
from deepgrad.backend import SimdTensorBackend, c_float
from deepgrad.utils import buffer_from

_zero_buf = {}

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
                
                # Then apply SGD update in-place
                SimdTensorBackend.sgd_update_inplace(
                    buffer_from(param.data),
                    buffer_from(param.grad),
                    len(param.data),
                    c_float(self.lr)
                )

    def zero_grad(self):
        for param in self.parameters:
            if param.requires_grad and param.grad:
                n = len(param.grad)
                if n not in _zero_buf:
                    _zero_buf[n] = array.array('f', [0.0] * n)
                param.grad[:] = _zero_buf[n]
