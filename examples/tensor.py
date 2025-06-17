import ctypes
import os
import array
from functools import lru_cache

# Set up C library interface
SimdTensorBackend = ctypes.cdll.LoadLibrary(os.path.abspath("../build/libsimd_tensor_backend.so"))

# Define types
c_float_p = ctypes.POINTER(ctypes.c_float)
c_size_t_p = ctypes.POINTER(ctypes.c_size_t)
c_float = ctypes.c_float
c_size_t = ctypes.c_size_t
c_bool = ctypes.c_bool
c_int = ctypes.c_int

# Define function signatures
function_signatures = {
    'tensor_ops_init': ([], c_int),

    'sanitize_gradients': ([c_float_p, c_size_t], None),
    'sgd_update_inplace': ([c_float_p, c_float_p, c_size_t, c_float_p], None),

    # Basic tensor operations
    **{f'tensor_{op}': ([c_float_p, c_float_p, c_float_p, c_size_t, c_size_t], None)
       for op in ['add', 'sub', 'mul', 'div']},

    # Gradients for tensor operations
    **{f'tensor_{op}_grad': ([c_float_p, c_float_p, c_float_p, c_float_p, c_float_p, c_size_t, c_size_t], None)
       for op in ['add', 'sub', 'mul', 'div']},
       
    'tensor_relu': ([c_float_p, c_float_p, c_size_t], None),
    'tensor_relu_backward': ([c_float_p, c_float_p, c_float_p, c_size_t], None),

    'tensor_matmul': ( [c_int, c_float_p, c_float_p, c_float_p, c_float_p,
                        c_float_p, c_size_t, c_size_t, c_size_t,c_size_t, c_bool], None),
 
    'tensor_softmax_ce': ([c_float_p, c_float_p, c_float_p, c_float_p,
                           c_float_p, c_float_p, c_size_t, c_size_t], None),

    'tensor_sum': ([c_float_p, c_size_t], c_float),
    'tensor_mean': ([c_float_p, c_size_t], c_float),
    
    'tensor_broadcast_row': ([c_float_p, c_float_p, c_size_t, c_size_t], None),
    'tensor_broadcast_col': ([c_float_p, c_float_p, c_size_t, c_size_t], None),
    'tensor_unbroadcast_sum_axes': ([c_float_p, c_float_p, c_size_t_p, c_size_t_p, c_size_t_p,
                                      c_size_t, c_size_t, c_size_t], None),

    'tensor_add_inplace': ([c_float_p, c_float_p, c_size_t], None),
    'tensor_fill_inplace': ([c_float_p, c_float, c_size_t], None),
    'zero_float_array': ([c_float_p, c_size_t], None),

}

# Set function signatures
for func_name, (argtypes, restype) in function_signatures.items():
    func = getattr(SimdTensorBackend, func_name)
    func.argtypes = argtypes
    func.restype = restype

OPS = {
    'add': ('tensor_add', 'tensor_add_grad'),
    'sub': ('tensor_sub', 'tensor_sub_grad'),
    'mul': ('tensor_mul', 'tensor_mul_grad'),
    'div': ('tensor_div', 'tensor_div_grad'),
    # Add more ops here as needed
}

_broadcast_cache = {}

def get_broadcast_cache_key(data, from_shape, to_shape):
    return (id(data), from_shape, to_shape)

_zero_buffer_pool = {}

def get_zero_buffer(size, shared=False):
    if size not in _zero_buffer_pool:
        _zero_buffer_pool[size] = array.array('f', [0.0] * size)
    else:
        SimdTensorBackend.zero_float_array(
            (c_float * size).from_buffer(_zero_buffer_pool[size]),
            size
        )
    
        if shared:
            return _zero_buffer_pool[size]

    # Return a copy to ensure isolation (still float32)
    return array.array('f', _zero_buffer_pool[size])

def buffer_from(g):
    return (c_float * len(g)).from_buffer(g)

class Tensor:
    def __init__(self, data, requires_grad=False, shape=None):
        self.data = data
        self.requires_grad = requires_grad
        self._grad = None  # do NOT allocate grad memory here
        self._backward = None
        self._prev = []

        if shape is not None:
            self.shape = shape
        else:
            self.shape = (len(self.data),)
            
        expected_size = 1
        if self.shape:
            for dim in self.shape:
                expected_size *= dim
        else:
            expected_size = 1

        assert len(self.data) == expected_size, f"Shape {self.shape} incompatible with data length {len(self.data)}"

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
                    t._grad = get_zero_buffer(len(t.data), shared=False)
                else:
                    SimdTensorBackend.zero_float_array(
                        buffer_from(t._grad),
                        len(t._grad)
                    )
        
        if self.shape != (1,) and not (len(self.shape) == 0 or self.shape == ()):
            raise RuntimeError(
                f"Cannot call backward on non-scalar tensor with shape {self.shape}. "
                "Call `.sum().backward()` or pass an explicit gradient instead."
            )

        SimdTensorBackend.tensor_fill_inplace(
            buffer_from(self.grad),
            c_float(1.0),
            c_size_t(len(self.grad)))
        
        for t in reversed(topo):
            if t._backward is not None:
                t._backward()
                # t.sanitize_gradients()
                
        shape = self.shape
        ndim = len(shape)
        strides = [1] * ndim
        for i in reversed(range(ndim - 1)):
            strides[i] = strides[i + 1] * shape[i + 1]
        self.strides = tuple(strides)

    def __getstate__(self):
        state = self.__dict__.copy()

        if 'data' in state and isinstance(state['data'], array.array):
            state['data'] = list(state['data'])

        if '_grad' in state and state['_grad'] is not None:
            if isinstance(state['_grad'], array.array):
                state['_grad'] = list(state['_grad'])

        return state

    def __setstate__(self, state):
        if 'data' in state and isinstance(state['data'], list):
            state['data'] = array.array('f', state['data'])

        if '_grad' in state and state['_grad'] is not None:
            if isinstance(state['_grad'], list):
                state['_grad'] = array.array('f', state['_grad'])

        self.__dict__.update(state)

    def __len__(self):
        return self.shape[0] if self.shape else 1

    @property
    def grad(self):
        if self.requires_grad:
            if self._grad is None:
                self._grad = get_zero_buffer(len(self.data), shared=False)  # fresh writable buffer
            return self._grad
        return None

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
        # Fast path: no-op broadcast
        if from_shape == to_shape:
            return data

        # Fast path: scalar to anything
        if from_shape == (1,) or len(from_shape) == 0:
            size = 1
            for dim in to_shape:
                size *= dim
            return array.array('f', data * size)

        key = (id(data), from_shape, to_shape)
        if key in _broadcast_cache:
            return _broadcast_cache[key]

        # Only 2D broadcast supported via C
        if len(from_shape) == 2 and len(to_shape) == 2:
            B, N = to_shape
            result = get_zero_buffer(B * N, shared=False)

            if from_shape[0] == 1 and from_shape[1] == N:
                SimdTensorBackend.tensor_broadcast_row(
                    buffer_from(data),
                    buffer_from(result),
                    B, N
                )
            elif from_shape[1] == 1 and from_shape[0] == B:
                SimdTensorBackend.tensor_broadcast_col(
                    buffer_from(data),
                    buffer_from(result),
                    B, N
                )
            else:
                raise NotImplementedError(f"Unsupported 2D broadcast from {from_shape} to {to_shape}")

            _broadcast_cache[key] = result
            return result

        raise NotImplementedError(f"Unsupported broadcast from {from_shape} to {to_shape}")

    def _unbroadcast_grad(self, grad, shape):
        grad_shape = self.shape
        ndim = len(grad_shape)

        if len(shape) != ndim:
            # Right-align shapes for broadcasting
            shape = (1,) * (ndim - len(shape)) + shape

        grad_arr = array.array('f', grad)
        
        # Calculate sizes directly without reduce
        grad_sz = 1
        for dim in grad_shape:
            grad_sz *= dim
            
        out_sz = 1
        for dim in shape:
            out_sz *= dim
            
        out_arr = get_zero_buffer(out_sz, shared=False)

        # === Inlined stride computation ===
        strides_grad = [1] * ndim
        for i in reversed(range(ndim - 1)):
            strides_grad[i] = strides_grad[i + 1] * grad_shape[i + 1]

        strides_out = [1] * ndim
        for i in reversed(range(ndim - 1)):
            strides_out[i] = strides_out[i + 1] * shape[i + 1]

        # Convert to ctypes arrays
        c_grad = (c_float * grad_sz).from_buffer(grad_arr)
        c_out = (c_float * out_sz).from_buffer(out_arr)
        c_shape_out = (c_size_t * ndim)(*shape)
        c_strides_grad = (c_size_t * ndim)(*strides_grad)
        c_strides_out = (c_size_t * ndim)(*strides_out)

        # Call C function
        SimdTensorBackend.tensor_unbroadcast_sum_axes(
            c_grad, c_out, c_shape_out,
            c_strides_grad, c_strides_out,
            ndim, grad_sz, out_sz
        )

        return out_arr
    
    def _apply_op(self, other, op_name, grad_fn_name):
        if not isinstance(other, Tensor):
            other = Tensor([other], shape=(1,), requires_grad=False)

        assert self.data.typecode == 'f'
        assert other.data.typecode == 'f'

        # Determine output shape
        out_shape = Tensor._compute_broadcast_shape(self.shape, other.shape)  # keep it

        # Broadcast data
        a_broadcasted = self.data if self.shape == out_shape else self._broadcast_data(self.data, self.shape, out_shape)
        b_broadcasted = other.data if other.shape == out_shape else self._broadcast_data(other.data, other.shape, out_shape)

        # Prepare output buffer
        out_size = 1
        for dim in out_shape:
            out_size *= dim
        out_data = get_zero_buffer(out_size, shared=False)

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
            buffer_from(self.data) if len(a_broadcasted) == len(self.data) else buffer_from(a_broadcasted),
            buffer_from(out_data) if len(b_broadcasted) == len(other.data) else buffer_from(b_broadcasted),
            buffer_from(out_data),
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

                self_grad = get_zero_buffer(len(a_broadcasted), shared=False)
                other_grad = get_zero_buffer(len(b_broadcasted), shared=False)

                grad_fn = getattr(SimdTensorBackend, grad_fn_name)

                if use_batch_cached:
                    grad_fn(
                        buffer_from(out_grad),
                        buffer_from(a_broadcasted),
                        buffer_from(b_broadcasted),
                        buffer_from(self_grad),
                        buffer_from(other_grad),
                        n,
                        batch_size_cached
                    )
                else:
                    grad_fn(
                        buffer_from(out_grad),
                        buffer_from(a_broadcasted),
                        buffer_from(b_broadcasted),
                        buffer_from(self_grad),
                        buffer_from(other_grad),
                        n
                    )

                if self.requires_grad:
                    self_grad = self._unbroadcast_grad(self_grad, self.shape)
                    SimdTensorBackend.tensor_add_inplace(
                        buffer_from(self.grad),
                        buffer_from(self_grad),
                        len(self.grad)
                    )

                if other.requires_grad:
                    other_grad = self._unbroadcast_grad(other_grad, other.shape)
                    SimdTensorBackend.tensor_add_inplace(
                        buffer_from(other.grad),
                        buffer_from(other_grad),
                        len(other.grad)
                    )

            out._backward = _backward
            out._prev = [self, other]

        return out

    def binary_op(self, other, op_name):
        if op_name not in OPS:
            raise ValueError(f"Unsupported op: {op_name}")
        forward_fn, backward_fn = OPS[op_name]
        return self._apply_op(other, forward_fn, backward_fn)

    def __add__(self, other): return self.binary_op(other, 'add')
    def __radd__(self, other): return self.__add__(other)

    def __sub__(self, other): return self.binary_op(other, 'sub')
    def __rsub__(self, other): return Tensor(other, requires_grad=False).__sub__(self)

    def __mul__(self, other): return self.binary_op(other, 'mul')
    def __rmul__(self, other): return self.__mul__(other)

    def __truediv__(self, other): return self.binary_op(other, 'div')
    def __rtruediv__(self, other): return Tensor(other, requires_grad=False).__truediv__(self)

    def matmul(self, other):
        assert isinstance(other, Tensor), "Operand must be a Tensor"

        s1, s2 = self.shape, other.shape

        # Check dimensions: support either 2D @ 2D, or batched (3D) @ 2D or 2D @ batched (3D)
        if len(s1) == 2 and len(s2) == 2:
            # Normal 2D matmul
            M, K = s1
            K2, N = s2
            assert K == K2, f"Incompatible matmul shapes {s1} and {s2}"

            batch = 1
            batch_self = False
            batch_other = False

        elif len(s1) == 3 and len(s2) == 2:
            # Batched self, 2D other (broadcast other)
            B, M, K = s1
            K2, N = s2
            assert K == K2, f"Incompatible matmul shapes {s1} and {s2}"
            batch = B
            batch_self = True
            batch_other = False

        elif len(s1) == 2 and len(s2) == 3:
            # 2D self, batched other (broadcast self)
            M, K = s1
            B, K2, N = s2
            assert K == K2, f"Incompatible matmul shapes {s1} and {s2}"
            batch = B
            batch_self = False
            batch_other = True

        else:
            raise NotImplementedError(f"Unsupported shapes for matmul: {s1} @ {s2}")

        out_shape = (batch, M, N) if batch > 1 else (M, N)
        out_size = batch * M * N if batch > 1 else M * N
        out_data = get_zero_buffer(out_size, shared=True)

        # If broadcasting needed, do not replicate buffers, just pass original pointers.
        # batch = batch size to lib
        SimdTensorBackend.tensor_matmul(
            0,  # MATMUL_FORWARD
            buffer_from(self.data),
            buffer_from(other.data),
            None,
            (c_float * out_size).from_buffer(out_data),
            None,
            batch,
            M, K, N,
            False
        )

        out = Tensor(array.array('f', out_data), requires_grad=self.requires_grad or other.requires_grad, shape=out_shape)

        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return

                grad_out_ptr = buffer_from(out.grad)
                grad_A_ptr = buffer_from(self.grad) if self.requires_grad else None
                grad_B_ptr = buffer_from(other.grad) if other.requires_grad else None

                SimdTensorBackend.tensor_matmul(
                    1,  # MATMUL_BACKWARD
                    buffer_from(self.data),
                    buffer_from(other.data),
                    grad_out_ptr,
                    grad_A_ptr,
                    grad_B_ptr,
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
        loss_data = get_zero_buffer((B), shared=False)
        grad_input = get_zero_buffer((B * C), shared=True)
        probs_data = get_zero_buffer((B * C), shared=True)

        # Forward pass: no grad_loss in the first call
        SimdTensorBackend.tensor_softmax_ce(
            buffer_from(self.data),
            buffer_from(target.data),
            None,  # grad_loss is NULL in the forward pass
            (c_float * B).from_buffer(loss_data),
            buffer_from(grad_input),
            buffer_from(probs_data),
            B,
            C
        )

        loss = Tensor(loss_data, requires_grad=self.requires_grad, shape=(B,))
        probs = Tensor(probs_data, shape=(B, C))  # For debugging or further use

        if loss.requires_grad:
            def _backward():
                if self.requires_grad:
                    # Call the fused function again, this time with grad_loss from next layer
                    SimdTensorBackend.tensor_softmax_ce(
                        buffer_from(self.data),
                        buffer_from(target.data),
                        buffer_from(loss.grad),  # Use upstream gradient
                        (c_float * B).from_buffer(loss_data),               # Losses can be reused
                        buffer_from(self.grad),  # Output gradient
                        None,  # probs_out not needed in backward
                        B,
                        C
                    )

            loss._backward = _backward
            loss._prev = [self, target]

        return loss.mean()

    def relu(self):
        out_data = get_zero_buffer(len(self.data), shared=False)

        SimdTensorBackend.tensor_relu(
            buffer_from(self.data),
            buffer_from(out_data),
            len(self.data)
        )
        
        out = Tensor(out_data, requires_grad=self.requires_grad, shape=self.shape)

        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                
                if self.requires_grad:
                    SimdTensorBackend.tensor_relu_backward(
                        buffer_from(out.grad),
                        buffer_from(self.data),
                        buffer_from(self.grad),
                        len(self.data)
                    )

            out._backward = _backward
            out._prev = [self]

        return out
    
    def mean(self):
        result = SimdTensorBackend.tensor_mean(buffer_from(self.data), len(self.data))
        out_data = get_zero_buffer(1, shared=False)
        out_data[0] = result
        out = Tensor(out_data, requires_grad=self.requires_grad)
        
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    grad_val = out.grad[0] / len(self.data)

                    # grad_array = array.array('f', [grad_val] * len(self.grad))
                    grad_array = get_zero_buffer(len(self.grad), shared=False)
                    for i in range(len(grad_array)):
                        grad_array[i] = grad_val

                    SimdTensorBackend.tensor_add_inplace(
                        buffer_from(self.grad),
                        buffer_from(grad_array),
                        len(self.grad)
                    )

            out._backward = _backward
            out._prev = [self]
        
        return out

    def sum(self):
        result = SimdTensorBackend.tensor_sum(buffer_from(self.data), len(self.data))
        out_data = get_zero_buffer(1, shared=False)
        out_data[0] = result
        out = Tensor(out_data, requires_grad=self.requires_grad)

        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    grad_val = out.grad[0]

                # grad_array = array.array('f', [grad_val] * len(self.grad))

                grad_array = get_zero_buffer(len(self.grad), shared=False)
                for i in range(len(grad_array)):
                    grad_array[i] = grad_val

                    SimdTensorBackend.tensor_add_inplace(
                        buffer_from(self.grad),
                        buffer_from(grad_array),
                        len(self.grad)
                    )

            out._backward = _backward
            out._prev = [self]

        return out

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={list(self.data)}, grad={list(self.grad) if self.grad else None})"
    
def _init_backend():
    result = SimdTensorBackend.tensor_ops_init()
    if result != 0:
        raise RuntimeError("tensor_ops_init failed (AVX2 unsupported?)")
    
_init_backend()