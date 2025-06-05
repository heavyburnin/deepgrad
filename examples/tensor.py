import ctypes
import os
import array
import math
from functools import lru_cache

# Set up C library interface
lib = ctypes.cdll.LoadLibrary(os.path.abspath("../build/libsimd_tensor_backend.so"))
c_float_p = ctypes.POINTER(ctypes.c_float)
c_size_t = ctypes.c_size_t

lib.tensor_ops_init.restype = ctypes.c_int
lib.tensor_ops_init.argtypes = []
if lib.tensor_ops_init() != 0:
    raise RuntimeError("Failed to initialize SIMD tensor backend (AVX2 may be missing)")

lib.sanitize_gradients.argtypes = [c_float_p, c_size_t]
lib.sgd_update_inplace.argtypes = [c_float_p, c_float_p, c_size_t, ctypes.c_float]
lib.tensor_add.argtypes = [c_float_p, c_float_p, c_float_p, c_size_t]
lib.tensor_sub.argtypes = [c_float_p, c_float_p, c_float_p, c_size_t]
lib.tensor_mul.argtypes = [c_float_p, c_float_p, c_float_p, c_size_t]
lib.tensor_div.argtypes = [c_float_p, c_float_p, c_float_p, c_size_t]
lib.tensor_relu.argtypes = [c_float_p, c_float_p, c_size_t]
lib.tensor_matmul_batch_jit.argtypes = [c_float_p, c_float_p, c_float_p, c_size_t, c_size_t, c_size_t, c_size_t]
lib.tensor_op_jit_softmax_ce_with_probs.argtypes = [c_float_p, c_float_p, c_float_p, c_float_p, c_size_t, c_size_t]
lib.tensor_exp.argtypes = [c_float_p, c_float_p, c_size_t]
lib.tensor_sum.restype = ctypes.c_float
lib.tensor_sum.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
lib.tensor_mean.restype = ctypes.c_float
lib.tensor_mean.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
lib.tensor_broadcast_row.argtypes = [c_float_p, c_float_p, c_size_t, c_size_t]
lib.tensor_broadcast_col.argtypes = [c_float_p, c_float_p, c_size_t, c_size_t]
lib.tensor_transpose_jit.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,  # ndim
    ctypes.c_size_t,  # B
    ctypes.c_size_t,  # M
    ctypes.c_size_t   # N
]

class Tensor:
    def __init__(self, data, requires_grad=False, shape=None):
        if isinstance(data, array.array) and data.typecode == 'f':
            self.data = data
        elif isinstance(data, array.array):
            self.data = array.array('f', data)
        elif isinstance(data, list):
            self.data = array.array('f', data)
        else:
            self.data = array.array('f', [float(data)])

        self.requires_grad = requires_grad
        self.grad = array.array('f', [0.0] * len(self.data)) if requires_grad else None
        self._backward = lambda: None
        self._prev = []

        if shape is not None:
            self.shape = shape
        else:
            self.shape = (len(self.data),)
            
        # Validate shape matches data length
        expected_size = math.prod(self.shape) if self.shape else 1
        assert len(self.data) == expected_size, f"Shape {self.shape} incompatible with data length {len(self.data)}"

    def __len__(self):
        return self.shape[0] if self.shape else 1

    def zero_grad(self):
        if self.requires_grad:
            for i in range(len(self.grad)):
                self.grad[i] = 0.0

    def sanitize_gradients(self):
        if self.grad is not None:
            # Convert list to array.array first if it's a list
            if isinstance(self.grad, list):
                self.grad = array.array('f', self.grad)
                
            # Now create the buffer from the array
            lib.sanitize_gradients(
                (ctypes.c_float * len(self.grad)).from_buffer(self.grad),
                len(self.grad)
            )
        return self

    def backward(self):
        if self.grad is None:
            self.grad = array.array('f', [0.0] * len(self.data))
        
        # Set initial gradient to 1.0
        for i in range(len(self.grad)):
            self.grad[i] = 1.0
            
        # Build computational graph
        visited = set()
        topo = []

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for p in t._prev:
                    build_topo(p)
                topo.append(t)

        build_topo(self)
        
        # Backward pass
        for t in reversed(topo):
            t._backward()
            # Sanitize gradients after each backward step
            if t.grad is not None:
                t.sanitize_gradients()

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_broadcast_shape(shape1, shape2):
        """Compute output shape for broadcasting with caching for performance."""
        if len(shape1) != len(shape2):
            raise ValueError(f"Only same-rank tensors supported for broadcasting (got {shape1} and {shape2})")
            
        out_shape = []
        for dim1, dim2 in zip(shape1, shape2):
            if dim1 == dim2:
                out_shape.append(dim1)
            elif dim1 == 1:
                out_shape.append(dim2)
            elif dim2 == 1:
                out_shape.append(dim1)
            else:
                raise ValueError(f"Incompatible shapes for broadcasting: {shape1} and {shape2}")
                
        return tuple(out_shape)

    @staticmethod
    def _broadcast_data(data, from_shape, to_shape):
        """Broadcast data from one shape to another using C acceleration when possible."""
        if from_shape == to_shape:
            return data[:]

        # Handle 2D broadcasting patterns efficiently
        if len(from_shape) == 2 and len(to_shape) == 2:
            # Case: [1, N] -> [B, N]  (row repeat)
            if from_shape[0] == 1 and from_shape[1] == to_shape[1]:
                B, N = to_shape
                result = array.array('f', [0.0] * (B * N))
                lib.tensor_broadcast_row(
                    (ctypes.c_float * len(data)).from_buffer(data),
                    (ctypes.c_float * len(result)).from_buffer(result),
                    B, N
                )
                return result

            # Case: [B, 1] -> [B, N]  (column repeat)
            elif from_shape[1] == 1 and from_shape[0] == to_shape[0]:  # [B, 1] → [B, N]
                B, N = to_shape
                result = array.array('f', [0.0] * (B * N))
                lib.tensor_broadcast_col(
                    (ctypes.c_float * len(data)).from_buffer(data),
                    (ctypes.c_float * len(result)).from_buffer(result),
                    B, N
                )
                return result

        # Scalar to any shape
        if from_shape == (1,):
            return data * math.prod(to_shape)

        raise NotImplementedError(f"Unsupported broadcast from {from_shape} to {to_shape}")

    def _apply_op(self, other, op_name, grad_fn):
        if not isinstance(other, Tensor):
            other = Tensor([other], shape=(1,), requires_grad=False)

        # Determine output shape
        out_shape = self._compute_broadcast_shape(self.shape, other.shape)
        
        # Broadcast data
        a = self._broadcast_data(self.data, self.shape, out_shape)
        b = self._broadcast_data(other.data, other.shape, out_shape)

        # Prepare output buffer
        out_size = math.prod(out_shape)
        out_data = array.array('f', [0.0] * out_size)
        
        # Call C function
        getattr(lib, op_name)(
            (ctypes.c_float * len(a))(*a),
            (ctypes.c_float * len(b))(*b),
            (ctypes.c_float * len(out_data)).from_buffer(out_data),
            len(out_data)
        )

        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad, shape=out_shape)

        if out.requires_grad:
            def _backward():
                if self.requires_grad and isinstance(self.grad, list):
                    self.grad = array.array('f', self.grad)
                if other.requires_grad and isinstance(other.grad, list):
                    other.grad = array.array('f', other.grad)
                 
                if self.requires_grad:
                    # Optimize gradient accumulation
                    self_grad = self.grad
                    out_grad = out.grad
                    for i in range(len(self_grad)):
                        self_grad[i] += grad_fn(a[i], b[i], out_grad[i], 'left')

                if other.requires_grad:
                    other_grad = other.grad
                    out_grad = out.grad
                    for i in range(len(other_grad)):
                        other_grad[i] += grad_fn(a[i], b[i], out_grad[i], 'right')

            out._backward = _backward
            out._prev = [self, other]

        return out

    def __add__(self, other):
        return self._apply_op(other, 'tensor_add', lambda a, b, g, l: g)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._apply_op(other, 'tensor_sub', lambda a, b, g, l: g if l == 'left' else -g)

    def __rsub__(self, other):
        return Tensor(other, requires_grad=False).__sub__(self)

    def __mul__(self, other):
        return self._apply_op(other, 'tensor_mul', lambda a, b, g, l: g * (b if l == 'left' else a))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._apply_op(other, 'tensor_div', lambda a, b, g, l: g / b if l == 'left' else -g * a / (b * b))

    def __rtruediv__(self, other):
        return Tensor(other, requires_grad=False).__truediv__(self)

    def reshape(self, new_shape):
        size = math.prod(new_shape) if new_shape else 1
        assert size == len(self.data), f"Cannot reshape size {len(self.data)} to {new_shape}"
        
        out = Tensor(self.data[:], requires_grad=self.requires_grad, shape=new_shape)

        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    if isinstance(self.grad, list):
                        self.grad = array.array('f', self.grad)

                    for i in range(len(self.data)):
                        self.grad[i] += out.grad[i]

            out._backward = _backward
            out._prev = [self]

        return out

    def grad_tensor(self):
        if self.grad is None:
            return Tensor(array.array('f', [0.0] * len(self.data)), requires_grad=False, shape=self.shape)
        return Tensor(self.grad[:], requires_grad=False, shape=self.shape)

    def transpose(self):
        """Transpose tensor dimensions.
        - For 1D: Convert to column vector (N,) -> (N, 1)
        - For 2D: Swap dimensions (M, N) -> (N, M)
        - For 3D: Transpose last two dimensions (B, M, N) -> (B, N, M)
        """
        if len(self.shape) == 1:
            # 1D case: vector to column vector
            N = self.shape[0]
            out_shape = (N, 1)
            transposed_data = array.array('f', [0.0] * N)
            
            # Simple copy for 1D case
            for i in range(N):
                transposed_data[i] = self.data[i]
                
        elif len(self.shape) == 2:
            # 2D case: matrix transpose
            M, N = self.shape
            out_shape = (N, M)
            transposed_data = array.array('f', [0.0] * (M * N))
            
            # Call C implementation for 2D transpose
            lib.tensor_transpose_jit(
                (ctypes.c_float * len(self.data)).from_buffer(self.data),
                (ctypes.c_float * len(transposed_data)).from_buffer(transposed_data),
                ctypes.c_size_t(2),  # ndim
                ctypes.c_size_t(1),  # B (unused for 2D)
                ctypes.c_size_t(M),
                ctypes.c_size_t(N)
            )
            
        elif len(self.shape) == 3:
            # 3D case: batch transpose
            B, M, N = self.shape
            out_shape = (B, N, M)
            transposed_data = array.array('f', [0.0] * (B * M * N))
            
            # Call C implementation for 3D batch transpose
            lib.tensor_transpose_jit(
                (ctypes.c_float * len(self.data)).from_buffer(self.data),
                (ctypes.c_float * len(transposed_data)).from_buffer(transposed_data),
                ctypes.c_size_t(3),  # ndim
                ctypes.c_size_t(B),
                ctypes.c_size_t(M),
                ctypes.c_size_t(N)
            )
        else:
            raise NotImplementedError(f"Transpose not implemented for {len(self.shape)}D tensors")

        out = Tensor(transposed_data, requires_grad=self.requires_grad, shape=out_shape)

        if out.requires_grad:
            def _backward():
                if self.requires_grad and out.grad:
                    if isinstance(self.grad, list):
                        self.grad = array.array('f', self.grad)
                    if len(self.shape) == 1:
                        # 1D case: simple copy
                        for i in range(len(self.grad)):
                            self.grad[i] += out.grad[i]
                            
                    elif len(self.shape) == 2:
                        # 2D case: use the same function for gradient (transpose is its own inverse)
                        M, N = self.shape
                        lib.tensor_transpose_jit(
                            (ctypes.c_float * len(out.grad))(*out.grad),
                            (ctypes.c_float * len(self.grad)).from_buffer(self.grad),
                            ctypes.c_size_t(2),  # ndim
                            ctypes.c_size_t(1),  # B (unused for 2D)
                            ctypes.c_size_t(N),  # Note: dimensions swapped for gradient
                            ctypes.c_size_t(M)
                        )
                        
                    elif len(self.shape) == 3:
                        # 3D case: use the same function for gradient
                        B, M, N = self.shape
                        lib.tensor_transpose_jit(
                            (ctypes.c_float * len(out.grad))(*out.grad),
                            (ctypes.c_float * len(self.grad)).from_buffer(self.grad),
                            ctypes.c_size_t(3),  # ndim
                            ctypes.c_size_t(B),
                            ctypes.c_size_t(N),  # Note: dimensions swapped for gradient
                            ctypes.c_size_t(M)
                        )

            out._backward = _backward
            out._prev = [self]

        return out

    def relu(self):
        out_data = array.array('f', [0.0] * len(self.data))

        lib.tensor_relu(
            (ctypes.c_float * len(self.data)).from_buffer(self.data),
            (ctypes.c_float * len(out_data)).from_buffer(out_data),
            len(self.data)
        )

        out = Tensor(out_data, requires_grad=self.requires_grad, shape=self.shape)

        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    if isinstance(self.grad, list):
                        self.grad = array.array('f', self.grad)
                    data = self.data
                    out_grad = out.grad
                    self_grad = self.grad
                    for i in range(len(self_grad)):
                        self_grad[i] += out_grad[i] if data[i] > 0 else 0.0

            out._backward = _backward
            out._prev = [self]

        return out

    def matmul(self, other):
        assert isinstance(other, Tensor), "Operand must be a Tensor"
        s1, s2 = self.shape, other.shape

        # Case: [B, M] @ [M, N] => [B, N]
        if len(s1) == 2 and len(s2) == 2:
            B, M = s1
            M2, N = s2
            assert M == M2, f"Incompatible matmul shapes {s1} and {s2}"

            out_data = array.array('f', [0.0] * (B * N))

            lib.tensor_matmul_batch_jit(
                (ctypes.c_float * len(self.data)).from_buffer(self.data),
                (ctypes.c_float * len(other.data)).from_buffer(other.data),
                (ctypes.c_float * len(out_data)).from_buffer(out_data),
                B, 1, M, N
            )

            out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad, shape=(B, N))

            if out.requires_grad:
                def _backward():
                    if self.requires_grad:
                        if isinstance(self.grad, list):
                            self.grad = array.array('f', self.grad)

                        b_T = other.transpose()
                        dA = out.grad_tensor().matmul(b_T)
                        for i in range(len(self.grad)):
                            self.grad[i] += dA.data[i]

                    if other.requires_grad:
                        if isinstance(other.grad, list):
                            other.grad = array.array('f', other.grad)
                        a_T = self.transpose()
                        dB = a_T.matmul(out.grad_tensor())
                        for i in range(len(other.grad)):
                            other.grad[i] += dB.data[i]

                out._backward = _backward
                out._prev = [self, other]

            return out

        # Case: [B, M, K] @ [B, K, N] => [B, M, N]
        elif len(s1) == 3 and len(s2) == 3:
            B1, M, K1 = s1
            B2, K2, N = s2
            assert B1 == B2 and K1 == K2, f"Incompatible batched matmul shapes {s1} and {s2}"

            out_data = array.array('f', [0.0] * (B1 * M * N))

            lib.tensor_matmul_batch_jit(
                (ctypes.c_float * len(self.data)).from_buffer(self.data),
                (ctypes.c_float * len(other.data)).from_buffer(other.data),
                (ctypes.c_float * len(out_data)).from_buffer(out_data),
                B1, M, K1, N
            )

            out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad, shape=(B1, M, N))

            if out.requires_grad:
                def _backward():
                    if self.requires_grad:
                        if isinstance(self.grad, list):
                            self.grad = array.array('f', self.grad)
                        b_T = other.transpose()
                        dA = out.grad_tensor().matmul(b_T)
                        for i in range(len(self.grad)):
                            self.grad[i] += dA.data[i]

                    if other.requires_grad:
                        if isinstance(other.grad, list):
                            other.grad = array.array('f', other.grad)
                        a_T = self.transpose()
                        dB = a_T.matmul(out.grad_tensor())
                        for i in range(len(other.grad)):
                            other.grad[i] += dB.data[i]

                out._backward = _backward
                out._prev = [self, other]

            return out

        else:
            raise NotImplementedError(f"Unsupported shapes for matmul: {s1} @ {s2}")

    def cross_entropy(self, target):
        assert self.shape == target.shape, f"Shape mismatch: {self.shape} vs {target.shape}"
        B, C = self.shape
        loss_data = array.array('f', [0.0] * B)
        probs_data = array.array('f', [0.0] * (B * C))

        lib.tensor_op_jit_softmax_ce_with_probs(
            (ctypes.c_float * len(self.data)).from_buffer(self.data),
            (ctypes.c_float * len(target.data)).from_buffer(target.data),
            (ctypes.c_float * B).from_buffer(loss_data),
            (ctypes.c_float * (B * C)).from_buffer(probs_data),
            B, C
        )

        loss = Tensor(loss_data, requires_grad=self.requires_grad, shape=(B,))
        probs = Tensor(probs_data, shape=(B, C))  # only for backprop

        if loss.requires_grad:
            def _backward():
                if self.requires_grad:
                    if isinstance(self.grad, list):
                        self.grad = array.array('f', self.grad)
                    scale = loss.grad[0] if len(loss.grad) == 1 else None
                    for i in range(B * C):
                        grad_scale = scale if scale is not None else loss.grad[i // C]
                        self.grad[i] += (probs.data[i] - target.data[i]) * grad_scale

            loss._backward = _backward
            loss._prev = [self, target]

        return loss.mean()

    def mean(self):
        # Call C implementation for mean calculation
        result = lib.tensor_mean((ctypes.c_float * len(self.data)).from_buffer(self.data), len(self.data))

        # Create output tensor with the scalar result
        out = Tensor([result], requires_grad=self.requires_grad)
        
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    if isinstance(self.grad, list):
                        self.grad = array.array('f', self.grad)
                    grad_val = out.grad[0] / len(self.data)
                    for i in range(len(self.grad)):
                        self.grad[i] += grad_val
            
            out._backward = _backward
            out._prev = [self]
        
        return out

    def sum(self):
        # Call C implementation for sum calculation
        result = lib.tensor_sum((ctypes.c_float * len(self.data)).from_buffer(self.data), len(self.data))

        # Create output tensor with the scalar result
        out = Tensor([result], requires_grad=self.requires_grad)
        
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    if isinstance(self.grad, list):
                        self.grad = array.array('f', self.grad)
                    grad_val = out.grad[0]
                    for i in range(len(self.grad)):
                        self.grad[i] += grad_val
            
            out._backward = _backward
            out._prev = [self]
        
        return out

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={list(self.data)}, grad={list(self.grad) if self.grad else None})"