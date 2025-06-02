# Enhanced Tensor class with batching support (no NumPy)
import ctypes
import os
import array

lib = ctypes.cdll.LoadLibrary(os.path.abspath("../build/libsimd_tensor_backend.so"))
c_float_p = ctypes.POINTER(ctypes.c_float)
c_size_t = ctypes.c_size_t

class Tensor:
    def __init__(self, data, requires_grad=False, shape=None):
        if isinstance(data, array.array):
            self.data = data
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

    def __len__(self):
        return self.shape[0] if len(self.shape) > 0 else 1

    def zero_grad(self):
        if self.requires_grad:
            self.grad = array.array('f', [0.0] * len(self.data))

    def backward(self):
        self.grad = array.array('f', [1.0] * len(self.data))
        visited = set()
        topo = []

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for p in t._prev:
                    build_topo(p)
                topo.append(t)

        build_topo(self)
        for t in reversed(topo):
            t._backward()

    def _apply_op(self, other, op_name, grad_fn):
        if not isinstance(other, Tensor):
            other = Tensor([other], shape=(1,), requires_grad=False)

        # Determine output shape
        shape1 = self.shape
        shape2 = other.shape

        assert len(shape1) == len(shape2), f"Only same-rank tensors supported for broadcasting (got {shape1} and {shape2})"

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
        
        out_shape = tuple(out_shape)  # normalize shape to tuple
        # Broadcast data manually
        def broadcast(data, from_shape, to_shape):
            if from_shape == to_shape:
                return data[:]
            elif from_shape[0] == 1 and from_shape[1] == to_shape[1]:  # row vector
                return data * to_shape[0]
            elif from_shape[1] == 1 and from_shape[0] == to_shape[0]:  # column vector
                return [v for v in data for _ in range(to_shape[1])]
            elif from_shape == (1,):
                return data * (to_shape[0] * to_shape[1])
            else:
                raise NotImplementedError(f"Unsupported broadcast from {from_shape} to {to_shape}")

        a = broadcast(self.data, self.shape, out_shape)
        b = broadcast(other.data, other.shape, out_shape)

        out_data = array.array('f', [0.0] * len(a))
        getattr(lib, op_name)(
            (ctypes.c_float * len(a))(*a),
            (ctypes.c_float * len(b))(*b),
            (ctypes.c_float * len(out_data)).from_buffer(out_data),
            len(out_data)
        )

        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad, shape=tuple(out_shape))

        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    self.grad[i] += grad_fn(a[i], b[i], out.grad[i], 'left')

            if other.requires_grad:
                for i in range(len(other.grad)):
                    other.grad[i] += grad_fn(a[i], b[i], out.grad[i], 'right')

        if out.requires_grad:
            out._backward = _backward
            out._prev = [self, other]

        return out

    def __add__(self, other):
        return self._apply_op(other, 'tensor_add', lambda a, b, g, l: g)

    def __sub__(self, other):
        return self._apply_op(other, 'tensor_sub', lambda a, b, g, l: g if l else -g)

    def __mul__(self, other):
        return self._apply_op(other, 'tensor_mul', lambda a, b, g, l: g * (b if l else a))

    def __truediv__(self, other):
        return self._apply_op(other, 'tensor_div', lambda a, b, g, l: g / (b if l else -a / (b * b)))

    def reshape(self, new_shape):
        size = 1
        for d in new_shape:
            size *= d
        assert size == len(self.data), f"Cannot reshape size {len(self.data)} to {new_shape}"
        out = Tensor(self.data[:], requires_grad=self.requires_grad, shape=new_shape)

        def _backward():
            if self.requires_grad:
                for i in range(len(out.grad)):
                    self.grad[i] += out.grad[i]

        if out.requires_grad:
            out._backward = _backward
            out._prev = [self]

        return out

    def grad_tensor(self):
        return Tensor(self.grad[:], requires_grad=False, shape=self.shape)

    def transpose(self):
        if self.shape is None:
            raise ValueError("Tensor shape is required for transpose")

        if len(self.shape) == 1:
            M, N = 1, self.shape[0]
        elif len(self.shape) == 2:
            M, N = self.shape
        else:
            raise NotImplementedError("Only 1D or 2D transpose is supported")

        transposed_data = array.array('f', [0.0] * (M * N))
        for i in range(M):
            for j in range(N):
                transposed_data[j * M + i] = self.data[i * N + j]

        out_shape = (N, M)
        out = Tensor(transposed_data, requires_grad=self.requires_grad, shape=out_shape)

        def _backward():
            if self.requires_grad and out.grad:
                for i in range(M):
                    for j in range(N):
                        self.grad[i * N + j] += out.grad[j * M + i]

        if out.requires_grad:
            out._backward = _backward
            out._prev = [self]

        return out
    
    def transpose_batch(self):
        assert len(self.shape) == 3, "transpose_batch only supports 3D tensors"
        B, M, N = self.shape
        out_data = array.array('f', [0.0] * (B * N * M))
        for b in range(B):
            for i in range(M):
                for j in range(N):
                    out_data[b * N * M + j * M + i] = self.data[b * M * N + i * N + j]
   
        out = Tensor(out_data, requires_grad=self.requires_grad, shape=(B, N, M))

        def _backward():
            if self.requires_grad and out.grad:
                for b in range(B):
                    for i in range(M):
                        for j in range(N):
                            self.grad[b * M * N + i * N + j] += out.grad[b * N * M + j * M + i]

        if out.requires_grad:
            out._backward = _backward
            out._prev = [self]

        return out

    lib.tensor_relu.argtypes = [c_float_p, c_float_p, ctypes.c_size_t]
    lib.tensor_relu.restype = None

    def relu(self):
        out_data = array.array('f', [0.0] * len(self.data))

        lib.tensor_relu(
            (ctypes.c_float * len(self.data))(*self.data),
            (ctypes.c_float * len(out_data)).from_buffer(out_data),
            len(self.data)
        )

        out = Tensor(out_data, requires_grad=self.requires_grad, shape=self.shape)

        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    self.grad[i] += out.grad[i] if self.data[i] > 0 else 0.0

        if out.requires_grad:
            out._backward = _backward
            out._prev = [self]

        return out
    
    # Register new function signature
    lib.tensor_matmul_batch_transposedB.argtypes = [
        c_float_p, c_float_p, c_float_p,
        c_size_t, c_size_t, c_size_t, c_size_t
    ]
    lib.tensor_matmul_batch_transposedB.restype = None

    def matmul_tra(self, other):
        assert isinstance(other, Tensor), "Operand must be a Tensor"
        s1, s2 = self.shape, other.shape

        # Case: [B, M] @ [M, N] => [B, N]
        if len(s1) == 2 and len(s2) == 2:
            B, M = s1
            M2, N = s2
            assert M == M2, f"Incompatible matmul shapes {s1} and {s2}"

            out_data = array.array('f', [0.0] * (B * N))
            B_T = other.transpose().data  # Transpose for compatibility

            lib.tensor_matmul_batch_transposedB(
                (ctypes.c_float * len(self.data))(*self.data),
                (ctypes.c_float * len(B_T))(*B_T),
                (ctypes.c_float * len(out_data)).from_buffer(out_data),
                B, 1, M, N
            )

            out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad, shape=(B, N))

            def _backward():
                if self.requires_grad:
                    b_T = other.transpose()
                    dA = out.grad_tensor().matmul(b_T)
                    for i in range(len(self.grad)):
                        self.grad[i] += dA.data[i]

                if other.requires_grad:
                    a_T = self.transpose()
                    dB = a_T.matmul(out.grad_tensor())
                    for i in range(len(other.grad)):
                        other.grad[i] += dB.data[i]

            if out.requires_grad:
                out._backward = _backward
                out._prev = [self, other]

            return out

        # Case: [B, M, K] @ [B, K, N] => [B, M, N]
        elif len(s1) == 3 and len(s2) == 3:
            B1, M, K1 = s1
            B2, K2, N = s2
            assert B1 == B2 and K1 == K2, f"Incompatible batched matmul shapes {s1} and {s2}"

            out_data = array.array('f', [0.0] * (B1 * M * N))
            B_T = other.transpose_batch().data  # Transpose for compatibility

            lib.tensor_matmul_batch_transposedB(
                (ctypes.c_float * len(self.data))(*self.data),
                (ctypes.c_float * len(B_T))(*B_T),
                (ctypes.c_float * len(out_data)).from_buffer(out_data),
                B1, M, K1, N
            )

            out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad, shape=(B1, M, N))

            def _backward():
                if self.requires_grad:
                    b_T = other.transpose_batch()
                    dA = out.grad_tensor().matmul(b_T)
                    for i in range(len(self.grad)):
                        self.grad[i] += dA.data[i]

                if other.requires_grad:
                    a_T = self.transpose_batch()
                    dB = a_T.matmul(out.grad_tensor())
                    for i in range(len(other.grad)):
                        other.grad[i] += dB.data[i]

            if out.requires_grad:
                out._backward = _backward
                out._prev = [self, other]

            return out

        else:
            raise NotImplementedError(f"Unsupported shapes for matmul: {s1} @ {s2}")
    
    lib.tensor_matmul_batch.argtypes = [
    c_float_p, c_float_p, c_float_p,
    c_size_t, c_size_t, c_size_t, c_size_t
    ]
    lib.tensor_matmul_batch.restype = None

    def matmul(self, other):
        assert isinstance(other, Tensor), "Operand must be a Tensor"
        s1, s2 = self.shape, other.shape

        # Case: [B, M] @ [M, N] => [B, N]
        if len(s1) == 2 and len(s2) == 2:
            B, M = s1
            M2, N = s2
            assert M == M2, f"Incompatible matmul shapes {s1} and {s2}"

            out_data = array.array('f', [0.0] * (B * N))

            lib.tensor_matmul_batch(
                (ctypes.c_float * len(self.data))(*self.data),
                (ctypes.c_float * len(other.data))(*other.data),
                (ctypes.c_float * len(out_data)).from_buffer(out_data),
                B, 1, M, N
            )

            out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad, shape=(B, N))

            def _backward():
                if self.requires_grad:
                    b_T = other.transpose()
                    dA = out.grad_tensor().matmul(b_T)
                    for i in range(len(self.grad)):
                        self.grad[i] += dA.data[i]

                if other.requires_grad:
                    a_T = self.transpose()
                    dB = a_T.matmul(out.grad_tensor())
                    for i in range(len(other.grad)):
                        other.grad[i] += dB.data[i]

            if out.requires_grad:
                out._backward = _backward
                out._prev = [self, other]

            return out

        # Case: [B, M, K] @ [B, K, N] => [B, M, N]
        elif len(s1) == 3 and len(s2) == 3:
            B1, M, K1 = s1
            B2, K2, N = s2
            assert B1 == B2 and K1 == K2, f"Incompatible batched matmul shapes {s1} and {s2}"

            out_data = array.array('f', [0.0] * (B1 * M * N))

            lib.tensor_matmul_batch(
                (ctypes.c_float * len(self.data))(*self.data),
                (ctypes.c_float * len(other.data))(*other.data),
                (ctypes.c_float * len(out_data)).from_buffer(out_data),
                B1, M, K1, N
            )

            out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad, shape=(B1, M, N))

            def _backward():
                if self.requires_grad:
                    b_T = other.transpose_batch()
                    dA = out.grad_tensor().matmul(b_T)
                    for i in range(len(self.grad)):
                        self.grad[i] += dA.data[i]

                if other.requires_grad:
                    a_T = self.transpose_batch()
                    dB = a_T.matmul(out.grad_tensor())
                    for i in range(len(other.grad)):
                        other.grad[i] += dB.data[i]

            if out.requires_grad:
                out._backward = _backward
                out._prev = [self, other]

            return out

        else:
            raise NotImplementedError(f"Unsupported shapes for matmul: {s1} @ {s2}")

    lib.tensor_op_jit_softmax_ce_with_probs.argtypes = [
        c_float_p,  # logits
        c_float_p,  # labels
        c_float_p,  # losses
        c_float_p,  # probs_out
        c_size_t,   # batch
        c_size_t    # class_count
    ]
    lib.tensor_op_jit_softmax_ce_with_probs.restype = None

    def cross_entropy(self, target):
        assert self.shape == target.shape, f"Shape mismatch: {self.shape} vs {target.shape}"
        B, C = self.shape
        loss_data = array.array('f', [0.0] * B)
        probs_data = array.array('f', [0.0] * (B * C))

        lib.tensor_op_jit_softmax_ce_with_probs(
            (ctypes.c_float * len(self.data))(*self.data),
            (ctypes.c_float * len(target.data))(*target.data),
            (ctypes.c_float * B).from_buffer(loss_data),
            (ctypes.c_float * (B * C)).from_buffer(probs_data),
            B, C
        )

        loss = Tensor(loss_data, requires_grad=self.requires_grad, shape=(B,))
        probs = Tensor(probs_data, shape=(B, C))  # only for backprop

        def _backward():
            if self.requires_grad:
                for i in range(B * C):
                    self.grad[i] += (probs.data[i] - target.data[i]) * \
                        (loss.grad[i // C] if len(loss.grad) > 1 else loss.grad[0])

        if loss.requires_grad:
            loss._backward = _backward
            loss._prev = [self, target]

        return loss.mean()

    def mean(self):
        total = sum(self.data)
        result = total / len(self.data)
        out = Tensor([result], requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    self.grad[i] += out.grad[0] / len(self.data)

        if out.requires_grad:
            out._backward = _backward
            out._prev = [self]

        return out

    def sum(self):
        total = sum(self.data)
        out = Tensor([total], requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    self.grad[i] += out.grad[0]

        if out.requires_grad:
            out._backward = _backward
            out._prev = [self]

        return out

    def exp(self):
        out_data = array.array('f', [0.0] * len(self.data))
        batch = 1
        stride = len(self.data)

        if len(self.shape) == 2:
            batch, stride = self.shape

        lib.tensor_op_jit_1in(lib.tensor_exp,
                              (ctypes.c_float * len(self.data))(*self.data),
                              (ctypes.c_float * len(out_data)).from_buffer(out_data),
                              batch, stride)

        out = Tensor(out_data, requires_grad=self.requires_grad, shape=self.shape)

        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    self.grad[i] += out.grad[i] * out.data[i]

        if out.requires_grad:
            out._backward = _backward
            out._prev = [self]

        return out
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={list(self.data)}, grad={list(self.grad) if self.grad else None})"