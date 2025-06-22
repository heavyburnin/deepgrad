from functools import lru_cache
from deepgrad.backend import SimdTensorBackend
from ctypes import c_float, c_size_t
from deepgrad.utils import get_broadcast_cache_key, get_broadcasted, set_broadcasted, broadcast_to_shape
from deepgrad.ops import get_op_names

class Tensor:
    def __init__(self, data, requires_grad=False, shape=None, size=None):
        self.data = data
        self.requires_grad = requires_grad
        self._grad = None
        self._backward = None
        self._prev = []

        if shape is not None:
            self.shape = shape
        else:
            # Infer 1D shape if not given
            if size is not None:
                self.shape = (size,)
            else:
                self.shape = (len(data),)

        # Compute expected size from shape
        expected_size = 1
        for dim in self.shape:
            expected_size *= dim

        # Determine actual size
        if size is not None:
            self.size = size
        else:
            try:
                self.size = len(data)
            except TypeError:
                raise ValueError("Must provide `size=` when using raw ctypes memory")

        # Validate size matches shape
        assert self.size == expected_size, f"Shape {self.shape} incompatible with data size {self.size}"

    def __getstate__(self):
        state = self.__dict__.copy()

        # Convert ctypes data buffer to list
        if 'data' in state and hasattr(state['data'], '__len__') and isinstance(state['data'], (c_float * len(state['data']))):
            state['data'] = [state['data'][i] for i in range(len(state['data']))]

        # Convert ctypes grad buffer to list
        if '_grad' in state and state['_grad'] is not None:
            if hasattr(state['_grad'], '__len__') and isinstance(state['_grad'], (c_float * len(state['_grad']))):
                state['_grad'] = [state['_grad'][i] for i in range(len(state['_grad']))]

        return state

    def __setstate__(self, state):
        # Convert list back to ctypes array for data
        if 'data' in state and isinstance(state['data'], list):
            state['data'] = (c_float * len(state['data']))(*state['data'])

        # Convert list back to ctypes array for grad
        if '_grad' in state and isinstance(state['_grad'], list):
            state['_grad'] = (c_float * len(state['_grad']))(*state['_grad'])

        self.__dict__.update(state)

    def __len__(self):
        return self.shape[0] if self.shape else 1

    @property
    def grad(self):
        if not self.requires_grad:
            return None

        if self._grad is None:
            self._grad = (c_float * self.self)()  # freshly allocated zero buffer
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_broadcast_shape(shape1, shape2):
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
        """
        Broadcasts the data from from_shape to to_shape using NumPy-style rules.
        """
        if from_shape == to_shape:
            return data

        key = get_broadcast_cache_key(data, from_shape, to_shape)
        cached = get_broadcasted(key)
        if cached is not None:
            return cached

        result = broadcast_to_shape(data, from_shape, to_shape)
        set_broadcasted(key, result)
        return result

    def _unbroadcast_grad(self, grad, shape):
        grad_shape = self.shape
        ndim = len(grad_shape)

        if len(shape) != ndim:
            shape = (1,) * (ndim - len(shape)) + shape

        # Assume grad is already a ctypes array
        grad_sz = 1
        for dim in grad_shape:
            grad_sz *= dim

        out_sz = 1
        for dim in shape:
            out_sz *= dim

        # Create ctypes output buffer
        out_arr = (c_float * out_sz)()

        # Compute strides
        strides_grad = [1] * ndim
        for i in reversed(range(ndim - 1)):
            strides_grad[i] = strides_grad[i + 1] * grad_shape[i + 1]

        strides_out = [1] * ndim
        for i in reversed(range(ndim - 1)):
            strides_out[i] = strides_out[i + 1] * shape[i + 1]

        # No more .from_buffer — all pure ctypes now
        c_shape_out = (c_size_t * ndim)(*shape)
        c_strides_grad = (c_size_t * ndim)(*strides_grad)
        c_strides_out = (c_size_t * ndim)(*strides_out)

        SimdTensorBackend.tensor_unbroadcast_sum_axes(
            grad,  # already a POINTER(c_float) or c_float array
            out_arr,
            c_shape_out,
            c_strides_grad,
            c_strides_out,
            ndim,
            grad_sz,
            out_sz
        )

        return out_arr

    def _apply_op(self, other, op_name, grad_fn_name):
        if not isinstance(other, Tensor):
            other = Tensor([other], shape=(1,), requires_grad=False)

        # Determine output shape
        out_shape = Tensor._compute_broadcast_shape(self.shape, other.shape)  # keep it

        # Broadcast data
        a_broadcasted = (
            self.data if self.shape == out_shape 
            else broadcast_to_shape(self.data, self.shape, out_shape, self.size)
        )

        b_broadcasted = (
            other.data if other.shape == out_shape 
            else broadcast_to_shape(other.data, other.shape, out_shape, other.size)
        )

        # Prepare output buffer
        out_size = 1
        for dim in out_shape:
            out_size *= dim
        out_data = (c_float * out_size)()

        # Compute batching
        if len(out_shape) > 1:
            batch_size = out_shape[0]
            n = 1
            for dim in out_shape[1:]:
                n *= dim
            use_batch = True
        else:
            batch_size = 1
            n = out_size
            use_batch = False

        # Call C op
        getattr(SimdTensorBackend, op_name)(
            a_broadcasted,
            b_broadcasted,
            out_data,
            n,
            batch_size if use_batch else 0  # only passed if batched
        )

        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad, shape=out_shape)

        # Cache broadcasted arrays only if needed
        if out.requires_grad:
            out._cached_a_broadcasted = a_broadcasted
            out._cached_b_broadcasted = b_broadcasted
            out._batch_info = (n, batch_size if use_batch else None)

            def _backward():
                out_grad = out.grad
                a_broadcasted = out._cached_a_broadcasted
                b_broadcasted = out._cached_b_broadcasted
                n, batch_size_cached = out._batch_info
                
                use_batch_cached = batch_size_cached is not None

                self_grad = (c_float * out_size)()
                other_grad = (c_float * out_size)()

                grad_fn = getattr(SimdTensorBackend, grad_fn_name)

                if use_batch_cached:
                    grad_fn(out_grad, a_broadcasted, b_broadcasted, self_grad, other_grad, n, batch_size_cached)
                else:
                    grad_fn(out_grad, a_broadcasted, b_broadcasted, self_grad, other_grad, n)

                if self.requires_grad:
                    self_grad = self._unbroadcast_grad(self_grad, self.shape)
                    SimdTensorBackend.tensor_add_inplace(self.grad, self_grad, self.size)

                if other.requires_grad:
                    other_grad = self._unbroadcast_grad(other_grad, other.shape)
                    SimdTensorBackend.tensor_add_inplace(other.grad, other_grad, other.size)

            out._backward = _backward
            out._prev = [self, other]

        return out
    
    def binary_op(self, other, op_name):
        forward_fn, backward_fn = get_op_names(op_name)
        return self._apply_op(other, forward_fn, backward_fn)
    
    def __add__(self, other): return self.binary_op(other, 'add')
    def __radd__(self, other): return self.__add__(other)

    def __sub__(self, other): return self.binary_op(other, 'sub')
    def __rsub__(self, other): return Tensor(other, requires_grad=False).__sub__(self)

    def __mul__(self, other): return self.binary_op(other, 'mul')
    def __rmul__(self, other): return self.__mul__(other)

    def __truediv__(self, other): return self.binary_op(other, 'div')
    def __rtruediv__(self, other): return Tensor(other, requires_grad=False).__truediv__(self)

    def __pow__(self, other): return self.binary_op(other, 'pow')
    def __rpow__(self, other): return Tensor(other, requires_grad=False).__pow__(self)

    def matmul(self, other):
        assert isinstance(other, Tensor), "Operand must be a Tensor"

        s1, s2 = self.shape, other.shape

        # Determine dimensions and batching
        if len(s1) == 2 and len(s2) == 2:
            M, K = s1
            K2, N = s2
            assert K == K2, f"Incompatible matmul shapes {s1} and {s2}"
            batch = 1
        elif len(s1) == 3 and len(s2) == 2:
            B, M, K = s1
            K2, N = s2
            assert K == K2, f"Incompatible matmul shapes {s1} and {s2}"
            batch = B
        elif len(s1) == 2 and len(s2) == 3:
            M, K = s1
            B, K2, N = s2
            assert K == K2, f"Incompatible matmul shapes {s1} and {s2}"
            batch = B
        else:
            raise NotImplementedError(f"Unsupported shapes for matmul: {s1} @ {s2}")

        # Output shape/size
        out_shape = (batch, M, N) if batch > 1 else (M, N)
        out_size = batch * M * N if batch > 1 else M * N

        out_data = (c_float * out_size)()

        # Forward pass
        SimdTensorBackend.tensor_matmul(
            0,  # MATMUL_FORWARD
            (c_float * self.size).from_buffer(self.data),
            (c_float * other.size).from_buffer(other.data),
            None,           # grad_out (unused)
            (c_float * out_size).from_buffer(out_data),
            None,           # grad_B (unused)
            batch,
            M, K, N,
            False
        )

        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad, shape=out_shape, size=out_size)

        # Backward setup
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return

                grad_A = self.grad if self.requires_grad else None
                grad_B = other.grad if other.requires_grad else None

                SimdTensorBackend.tensor_matmul(
                    1,  # MATMUL_BACKWARD
                    (c_float * self.size).from_buffer(self.data),
                    (c_float * other.size).from_buffer(other.data),
                    out.grad,
                    grad_A,
                    grad_B,
                    batch,
                    M, K, N,
                    True
                )

            out._backward = _backward
            out._prev = [self, other]

        return out

    def cross_entropy(self, target):
        assert self.shape == target.shape, f"Shape mismatch: {self.shape} vs {target.shape}"
        B, C = self.shape
        loss_data = (c_float * B)()
        grad_input = (c_float * (B * C))()
        probs_data = (c_float * (B * C))()

        SimdTensorBackend.tensor_softmax_ce(
            self.data,
            target.data,
            None,
            loss_data,
            grad_input,
            probs_data,
            B,
            C
        )

        loss = Tensor(loss_data, requires_grad=self.requires_grad, shape=(B,), size=B)

        if loss.requires_grad:
            def _backward():
                if self.requires_grad:
                    SimdTensorBackend.tensor_softmax_ce(
                        self.data,
                        target.data,
                        loss.grad,
                        loss_data,
                        self.grad,
                        None,
                        B,
                        C
                    )
            loss._backward = _backward
            loss._prev = [self, target]

        return loss.mean()

    def relu(self):
        out_data = (c_float * self.size)()

        SimdTensorBackend.tensor_relu(
            self.data,
            out_data,
            self.size
        )

        out = Tensor(out_data, requires_grad=self.requires_grad, shape=self.shape, size=self.size)

        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                if self.requires_grad:
                    SimdTensorBackend.tensor_relu_backward(
                        out.grad,
                        self.data,
                        self.grad,
                        self.size
                    )
            out._backward = _backward
            out._prev = [self]

        return out

    def mean(self):
        result = SimdTensorBackend.tensor_mean(self.data, self.size)
        out_data = (c_float * 1)()
        out_data[0] = result
        out = Tensor(out_data, requires_grad=self.requires_grad)

        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    grad_val = out.grad[0] / self.size
                    grad_array = (c_float * self.size)(*([grad_val] * self.size))
                    SimdTensorBackend.tensor_add_inplace(self.grad, grad_array, self.size)
            out._backward = _backward
            out._prev = [self]

        return out

    def sum(self):
        result = SimdTensorBackend.tensor_sum(self.data, self.size)
        out_data = (c_float * 1)()
        out_data[0] = result
        out = Tensor(out_data, requires_grad=self.requires_grad)

        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    grad_val = out.grad[0]
                    grad_array = (c_float * self.size)(*([grad_val] * self.size))
                    SimdTensorBackend.tensor_add_inplace(self.grad, grad_array, self.size)
            out._backward = _backward
            out._prev = [self]

        return out

    def backward(self):
        visited = set()
        topo = []

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for p in t._prev:
                    build_topo(p)
                topo.append(t)

        build_topo(self)

        for t in topo:
            if t.requires_grad:
                if t._grad is None:
                    t._grad = (c_float * t.size)()
                else:
                    SimdTensorBackend.zero_float_array(t._grad, t.size)

        if self.shape != (1,) and not (len(self.shape) == 0 or self.shape == ()): 
            raise RuntimeError(
                f"Cannot call backward on non-scalar tensor with shape {self.shape}. "
                "Call `.sum().backward()` or pass an explicit gradient instead."
            )

        SimdTensorBackend.tensor_fill_inplace(self.grad, c_float(1.0), c_size_t(len(self.grad)))

        for t in reversed(topo):
            if t._backward is not None:
                t._backward()

        # Set strides (optional, depends on rest of your code)
        shape = self.shape
        ndim = len(shape)
        strides = [1] * ndim
        for i in reversed(range(ndim - 1)):
            strides[i] = strides[i + 1] * shape[i + 1]
        self.strides = tuple(strides)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={[self.data[i] for i in range(len(self.data))]}, grad={[self.grad[i] for i in range(len(self.grad))] if self.grad else None})"

def _init_backend():
    result = SimdTensorBackend.tensor_ops_init()
    if result != 0:
        raise RuntimeError("tensor_ops_init failed (AVX2 unsupported?)")
    
_init_backend()