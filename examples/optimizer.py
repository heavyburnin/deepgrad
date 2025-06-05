import ctypes
import os
import array

lib = ctypes.cdll.LoadLibrary(os.path.abspath("../build/libsimd_tensor_backend.so"))
c_float_p = ctypes.POINTER(ctypes.c_float)
c_size_t = ctypes.c_size_t
lib.sgd_update_inplace.argtypes = [c_float_p, c_float_p, c_size_t, ctypes.c_float]

class SGD:
    def __init__(self, parameters, lr=0.001):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                # First sanitize gradients
                param.sanitize_gradients()
                
                # Convert lists to array.array if needed
                if isinstance(param.data, list):
                    param.data = array.array('f', param.data)
                if isinstance(param.grad, list):
                    param.grad = array.array('f', param.grad)
                    
                # Then apply SGD update in-place
                lib.sgd_update_inplace(
                    (ctypes.c_float * len(param.data)).from_buffer(param.data),
                    (ctypes.c_float * len(param.grad)).from_buffer(param.grad),
                    len(param.data),
                    ctypes.c_float(self.lr)
                )

    def zero_grad(self):
        for param in self.parameters:
            if param.requires_grad and param.grad:
                param.grad = [0.0] * len(param.grad)