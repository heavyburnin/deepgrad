from ctypes import c_float, c_size_t, c_bool, POINTER
from deepgrad.tensor import Tensor
from deepgrad.backend import SimdTensorBackend

c_float_p = POINTER(c_float)

class BatchNorm1D:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.dim = dim

        self.gamma = Tensor((c_float * dim)(*([1.0] * dim)),
                            shape=(1, dim), size=dim, requires_grad=True)
        self.beta = Tensor((c_float * dim)(*([0.0] * dim)),
                           shape=(1, dim), size=dim, requires_grad=True)

        self.running_mean = (c_float * dim)(*([0.0] * dim))
        self.running_var = (c_float * dim)(*([1.0] * dim))

    def __call__(self, x: Tensor):
        assert x.ndim == 2 and x.shape[1] == self.dim, f"Expected (B, {self.dim}), got {x.shape}"
        B, C = x.shape

        out_data = (c_float * x.size)()
        x_hat_data = (c_float * x.size)()

        SimdTensorBackend.batchnorm_forward_f32(
            x.data, out_data, x_hat_data,
            self.gamma.data, self.beta.data,
            self.running_mean, self.running_var,
            c_size_t(B), c_size_t(C), c_size_t(1), c_size_t(1),  # H=W=1
            c_float(self.eps), c_float(self.momentum),
            c_bool(self.training)
        )

        out = Tensor(out_data, shape=x.shape, size=x.size,
                     requires_grad=x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad)

        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return

                dx = (c_float * x.size)()
                dgamma = (c_float * C)()
                dbeta = (c_float * C)()

                SimdTensorBackend.batchnorm_backward_f32(
                    x.data, out.grad,
                    dx, dgamma, dbeta,
                    self.gamma.data,
                    c_size_t(B), c_size_t(C), c_size_t(1), c_size_t(1),
                    c_float(self.eps)
                )

                if x.grad is None:
                    x.grad = dx
                else:
                    for i in range(x.size):
                        x.grad[i] += dx[i]

                if self.gamma.requires_grad:
                    if self.gamma.grad is None:
                        self.gamma.grad = (c_float * C)()
                    for i in range(C):
                        self.gamma.grad[i] += dgamma[i]

                if self.beta.requires_grad:
                    if self.beta.grad is None:
                        self.beta.grad = (c_float * C)()
                    for i in range(C):
                        self.beta.grad[i] += dbeta[i]

            out._backward = _backward
            out._prev = [x, self.gamma, self.beta]

        return out

    def parameters(self):
        return [self.gamma, self.beta]

    def __getstate__(self):
        state = self.__dict__.copy()
        state['running_mean'] = list(self.running_mean)
        state['running_var'] = list(self.running_var)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.running_mean = (c_float * self.dim)(*state['running_mean'])
        self.running_var = (c_float * self.dim)(*state['running_var'])

class BatchNorm2D:
    def __init__(self, num_channels, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.num_channels = num_channels

        # Learnable parameters
        self.gamma = Tensor((c_float * num_channels)(*([1.0] * num_channels)),
                            shape=(1, num_channels, 1, 1), size=num_channels, requires_grad=True)
        self.beta = Tensor((c_float * num_channels)(*([0.0] * num_channels)),
                           shape=(1, num_channels, 1, 1), size=num_channels, requires_grad=True)

        # Running statistics (not learnable)
        self.running_mean = (c_float * num_channels)(*([0.0] * num_channels))
        self.running_var = (c_float * num_channels)(*([1.0] * num_channels))

    def __call__(self, x: Tensor):
        assert x.ndim == 4, f"Expected 4D input, got {x.shape}"
        B, C, H, W = x.shape

        # Allocate output and x_hat (needed for backward)
        out_data = (c_float * x.size)()
        x_hat_data = (c_float * x.size)()

        SimdTensorBackend.batchnorm_forward_f32(
            x.data, out_data, x_hat_data,
            self.gamma.data, self.beta.data,
            self.running_mean, self.running_var,
            c_size_t(B), c_size_t(C), c_size_t(H), c_size_t(W),
            c_float(self.eps), c_float(self.momentum),
            c_bool(self.training)
        )

        out = Tensor(out_data, shape=x.shape, size=x.size,
                     requires_grad=x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad)

        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return

                dx = (c_float * x.size)()
                dgamma = (c_float * C)()  # ✅ no need to pre-zero, C code accumulates
                dbeta  = (c_float * C)()

                SimdTensorBackend.batchnorm_backward_f32(
                    x.data, out.grad,
                    dx, dgamma, dbeta,
                    self.gamma.data,
                    c_size_t(B), c_size_t(C), c_size_t(H), c_size_t(W),
                    c_float(self.eps)
                )

                if self.gamma.requires_grad:
                    if self.gamma.grad is None:
                        self.gamma.grad = (c_float * C)()
                    SimdTensorBackend.accumulate_grad_avx(self.gamma.grad, dgamma, c_size_t(C))

                if self.beta.requires_grad:
                    if self.beta.grad is None:
                        self.beta.grad = (c_float * C)()
                    SimdTensorBackend.accumulate_grad_avx(self.beta.grad, dbeta, c_size_t(C))

                if x.grad is None:
                    x.grad = dx
                else:
                    SimdTensorBackend.accumulate_grad_avx(x.grad, dx, c_size_t(x.size))

            out._backward = _backward
            out._prev = [x, self.gamma, self.beta]

        return out

    def parameters(self):
        return [self.gamma, self.beta]

    def __getstate__(self):
        state = self.__dict__.copy()
        state['running_mean'] = list(self.running_mean)
        state['running_var'] = list(self.running_var)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.running_mean = (c_float * self.num_channels)(*state['running_mean'])
        self.running_var = (c_float * self.num_channels)(*state['running_var'])
