import ctypes
import os
import array

lib = ctypes.cdll.LoadLibrary(os.path.abspath("../build/libsimd_tensor_backend.so"))
lib.sgd_update_inplace.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_float]
lib.zero_float_array.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
_zero_buf = {}

class SGD:
    
    def __init__(self, parameters, lr=0.001):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
                
                # Then apply SGD update in-place
                lib.sgd_update_inplace(
                    (ctypes.c_float * len(param.data)).from_buffer(param.data),
                    (ctypes.c_float * len(param.grad)).from_buffer(param.grad),
                    len(param.data),
                    ctypes.c_float(self.lr)
                )

    def zero_grad_back(self):
        for param in self.parameters:
            if param.requires_grad and param.grad:
                param.grad = array.array('f', [0.0] * len(param.grad))

    def zero_grad_c(self):
        for param in self.parameters:
            if param.requires_grad and param.grad:
                ptr = (ctypes.c_float * len(param.grad)).from_buffer(param.grad)
                lib.zero_float_array(ptr, len(param.grad))

    def zero_grad(self):
        for param in self.parameters:
            if param.requires_grad and param.grad:
                n = len(param.grad)
                if n not in _zero_buf:
                    _zero_buf[n] = array.array('f', [0.0] * n)
                param.grad[:] = _zero_buf[n]
