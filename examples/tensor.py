import ctypes
import os
import array
from functools import lru_cache

# Set up C library interface
lib = ctypes.cdll.LoadLibrary(os.path.abspath("../build/libsimd_tensor_backend.so"))

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

    'tensor_add': ([c_float_p, c_float_p, c_float_p, c_size_t, c_size_t], None),
    'tensor_sub': ([c_float_p, c_float_p, c_float_p, c_size_t, c_size_t], None),
    'tensor_mul': ([c_float_p, c_float_p, c_float_p, c_size_t, c_size_t], None),
    'tensor_div': ([c_float_p, c_float_p, c_float_p, c_size_t, c_size_t], None),

    'tensor_add_grad': ([c_float_p, c_float_p, c_float_p, c_float_p, c_float_p, c_size_t, c_size_t], None),
    'tensor_sub_grad': ([c_float_p, c_float_p, c_float_p, c_float_p, c_float_p, c_size_t, c_size_t], None),
    'tensor_mul_grad': ([c_float_p, c_float_p, c_float_p, c_float_p, c_float_p, c_size_t, c_size_t], None),
    'tensor_div_grad': ([c_float_p, c_float_p, c_float_p, c_float_p, c_float_p, c_size_t, c_size_t], None),

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
    func = getattr(lib, func_name)
    func.argtypes = argtypes
    func.restype = restype

_buffer_pool = {}
def get_buffer(size):
    """Reuse or create a float32 buffer of the given size."""
    if size in _buffer_pool:
        return _buffer_pool[size]
    buf = array.array('f', [0.0] * size)
    _buffer_pool[size] = buf
    return buf

_broadcast_cache = {}
def get_broadcast_cache_key(data, from_shape, to_shape):
    return (id(data), from_shape, to_shape)

class Tensor:
    def __init__(self, data, requires_grad=False, shape=None):
        if isinstance(data, array.array):
            if data.typecode != 'f':
                raise TypeError(f"Expected float32 array ('f'), got typecode '{data.typecode}'")
            self.data = data
        else:
            self.data = array.array('f', data if isinstance(data, (list, tuple)) else [float(data)])

        self.requires_grad = requires_grad
        self.grad = array.array('f', [0.0] * len(self.data)) if requires_grad else None
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
                if t.grad is None:
                    t.grad = t.grad = array.array('f', [0.0] * len(t.data))
                lib.zero_float_array(
                    (c_float * len(t.grad)).from_buffer(t.grad),
                    len(t.grad)
                )
        
        lib.tensor_fill_inplace(
            (c_float * len(self.grad)).from_buffer(self.grad),
            c_float(1.0),
            c_size_t(len(self.grad)))
        
        for t in reversed(topo):
            if t._backward is not None:
                t._backward()

        def _compute_strides(shape):
            strides = [1] * len(shape)
            for i in reversed(range(len(shape) - 1)):
                strides[i] = strides[i + 1] * shape[i + 1]
            return tuple(strides)

        self.strides = _compute_strides(self.shape)

    def __getstate__(self):
        state = self.__dict__.copy()
        
        if 'data' in state and isinstance(state['data'], array.array):
            state['data'] = list(state['data'])
        
        if 'grad' in state and state['grad'] is not None:
            if isinstance(state['grad'], array.array):
                state['grad'] = list(state['grad'])
        return state

    def __setstate__(self, state):
        if 'data' in state and isinstance(state['data'], list):
            state['data'] = array.array('f', state['data'])
        
        if 'grad' in state and state['grad'] is not None:
            if isinstance(state['grad'], list):
                state['grad'] = array.array('f', state['grad'])
        
        self.__dict__.update(state)

    def __len__(self):
        return self.shape[0] if self.shape else 1

    def sanitize_gradients(self):
        if self.grad is not None:
            lib.sanitize_gradients(
                (ctypes.c_float * len(self.grad)).from_buffer(self.grad),
                len(self.grad)
            )
        return self

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
        key = get_broadcast_cache_key(data, from_shape, to_shape)
        if key in _broadcast_cache:
            return _broadcast_cache[key]
        
        if from_shape == to_shape:
            return data[:]

        # Handle 2D broadcasting patterns efficiently
        if len(from_shape) == 2 and len(to_shape) == 2:
            # Case: [1, N] -> [B, N]  (row repeat)
            if from_shape[0] == 1 and from_shape[1] == to_shape[1]:
                B, N = to_shape
                result = array.array('f', [0.0] * (B * N))
                lib.tensor_broadcast_row(
                    (c_float * len(data)).from_buffer(data),
                    (c_float * len(result)).from_buffer(result),
                    B, N
                )
                _broadcast_cache[key] = result
                return result

            # Case: [B, 1] -> [B, N]  (column repeat)
            elif from_shape[1] == 1 and from_shape[0] == to_shape[0]:  # [B, 1] → [B, N]
                B, N = to_shape
                result = get_buffer(B * N)
                lib.tensor_broadcast_col(
                    (c_float * len(data)).from_buffer(data),
                    (c_float * len(result)).from_buffer(result),
                    B, N
                )
                _broadcast_cache[key] = result
                return result

        # Scalar to any shape
        if from_shape == (1,):
            size = 1
            for dim in to_shape:
                size *= dim
            return data * size

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
            
        out_arr = get_buffer(out_sz)

        # Compute strides
        def compute_strides(shape):
            strides = [1] * len(shape)
            for i in reversed(range(len(shape) - 1)):
                strides[i] = strides[i + 1] * shape[i + 1]
            return strides

        strides_grad = compute_strides(grad_shape)
        strides_out = compute_strides(shape)

        # Convert to ctypes arrays
        c_grad = (c_float * grad_sz).from_buffer(grad_arr)
        c_out = (c_float * out_sz).from_buffer(out_arr)
        c_shape_out = (c_size_t * ndim)(*shape)
        c_strides_grad = (c_size_t * ndim)(*strides_grad)
        c_strides_out = (c_size_t * ndim)(*strides_out)

        # Call C function
        lib.tensor_unbroadcast_sum_axes(
            c_grad, c_out, c_shape_out,
            c_strides_grad, c_strides_out,
            ndim, grad_sz, out_sz
        )

        return out_arr
    
    def _apply_op(self, other, op_name, grad_fn_name):
        if not isinstance(other, Tensor):
            other = Tensor([other], shape=(1,), requires_grad=False)

        # Determine output shape
        out_shape = self._compute_broadcast_shape(self.shape, other.shape)

        # Broadcast data
        a_broadcasted = self._broadcast_data(self.data, self.shape, out_shape)
        b_broadcasted = self._broadcast_data(other.data, other.shape, out_shape)

        # Prepare output buffer
        out_size = 1
        for dim in out_shape:
            out_size *= dim
        out_data = array.array('f', [0.0] *out_size)

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
        getattr(lib, op_name)(
            (c_float * len(self.data)).from_buffer(self.data) if len(a_broadcasted) == len(self.data) else (c_float * len(a_broadcasted)).from_buffer(a_broadcasted),
            (c_float * len(out_data)).from_buffer(out_data) if len(b_broadcasted) == len(other.data) else (c_float * len(b_broadcasted)).from_buffer(b_broadcasted),
            (c_float * len(out_data)).from_buffer(out_data),
            n,
            batch_size if use_batch else 0  # only passed if batched
        )

        out = Tensor(array.array('f', out_data), requires_grad=self.requires_grad or other.requires_grad, shape=out_shape)

        # 🔽 Cache broadcasted arrays only if needed
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

                self_grad = array.array('f', [0.0] *  len(a_broadcasted))
                other_grad = array.array('f', [0.0] * len(b_broadcasted))

                grad_fn = getattr(lib, grad_fn_name)

                if use_batch_cached:
                    grad_fn(
                        (c_float * len(out_grad)).from_buffer(out_grad),
                        (c_float * len(a_broadcasted)).from_buffer(a_broadcasted),
                        (c_float * len(b_broadcasted)).from_buffer(b_broadcasted),
                        (c_float * len(self_grad)).from_buffer(self_grad),
                        (c_float * len(other_grad)).from_buffer(other_grad),
                        n,
                        batch_size_cached
                    )
                else:
                    grad_fn(
                        (c_float * len(out_grad)).from_buffer(out_grad),
                        (c_float * len(a_broadcasted)).from_buffer(a_broadcasted),
                        (c_float * len(b_broadcasted)).from_buffer(b_broadcasted),
                        (c_float * len(self_grad)).from_buffer(self_grad),
                        (c_float * len(other_grad)).from_buffer(other_grad),
                        n
                    )

                if self.requires_grad:
                    self_grad = self._unbroadcast_grad(self_grad, self.shape)
                    lib.tensor_add_inplace(
                        (c_float * len(self.grad)).from_buffer(self.grad),
                        (c_float * len(self_grad)).from_buffer(self_grad),
                        len(self.grad)
                    )

                if other.requires_grad:
                    other_grad = self._unbroadcast_grad(other_grad, other.shape)
                    lib.tensor_add_inplace(
                        (c_float * len(other.grad)).from_buffer(other.grad),
                        (c_float * len(other_grad)).from_buffer(other_grad),
                        len(other.grad)
                    )

            out._backward = _backward
            out._prev = [self, other]

        return out

    def __add__(self, other):
        return self._apply_op(other, 'tensor_add', 'tensor_add_grad')

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._apply_op(other, 'tensor_sub', 'tensor_sub_grad')

    def __rsub__(self, other):
        return Tensor(other, requires_grad=False).__sub__(self)

    def __mul__(self, other):
        return self._apply_op(other, 'tensor_mul', 'tensor_mul_grad')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._apply_op(other, 'tensor_div', 'tensor_div_grad')

    def __rtruediv__(self, other):
        return Tensor(other, requires_grad=False).__truediv__(self)

    def matmul(self, other):
        assert isinstance(other, Tensor), "Operand must be a Tensor"
        s1, s2 = self.shape, other.shape
        if len(s1) == 2 and len(s2) == 2:
            M, K = s1
            K2, N = s2
            assert K == K2, f"Incompatible matmul shapes {s1} and {s2}"
        
            out_data = get_buffer(M * N)

            lib.tensor_matmul(
                0,  # MATMUL_FORWARD
                (c_float * len(self.data)).from_buffer(self.data),
                (c_float * len(other.data)).from_buffer(other.data),
                None,
                (c_float * len(out_data)).from_buffer(out_data),
                None,
                1,  # batch = 1
                M, K, N,
                False
            )
            
            out = Tensor(array.array('f', out_data), requires_grad=self.requires_grad or other.requires_grad, shape=(M, N))
                
            if out.requires_grad:
                def _backward():
                    if out.grad is None:
                        return
                        
                    grad_out_ptr = (c_float * len(out.grad)).from_buffer(out.grad)
                    grad_A_ptr = (c_float * len(self.grad)).from_buffer(self.grad) if self.requires_grad else None
                    grad_B_ptr = (c_float * len(other.grad)).from_buffer(other.grad) if other.requires_grad else None

                    lib.tensor_matmul(
                        1,  # MATMUL_BACKWARD
                        (c_float * len(self.data)).from_buffer(self.data),
                        (c_float * len(other.data)).from_buffer(other.data),
                        grad_out_ptr,
                        grad_A_ptr,
                        grad_B_ptr,
                        1,  # batch
                        M, K, N,
                        True
                    )

                out._backward = _backward
                out._prev = [self, other]

            return out

        else:
            raise NotImplementedError(f"Unsupported shapes for matmul: {s1} @ {s2}")

    def cross_entropy(self, target):
        assert self.shape == target.shape, f"Shape mismatch: {self.shape} vs {target.shape}"
        B, C = self.shape
        loss_data = get_buffer(B)
        grad_input = get_buffer(B * C)
        probs_data = get_buffer(B * C)

        # Forward pass: no grad_loss in the first call
        lib.tensor_softmax_ce(
            (c_float * len(self.data)).from_buffer(self.data),
            (c_float * len(target.data)).from_buffer(target.data),
            None,  # grad_loss is NULL in the forward pass
            (c_float * B).from_buffer(loss_data),
            (c_float * len(grad_input)).from_buffer(grad_input),
            (c_float * len(probs_data)).from_buffer(probs_data),
            B,
            C
        )

        loss = Tensor(loss_data, requires_grad=self.requires_grad, shape=(B,))
        probs = Tensor(probs_data, shape=(B, C))  # For debugging or further use

        if loss.requires_grad:
            def _backward():
                if self.requires_grad:
                    # Call the fused function again, this time with grad_loss from next layer
                    lib.tensor_softmax_ce(
                        (c_float * len(self.data)).from_buffer(self.data),
                        (c_float * len(target.data)).from_buffer(target.data),
                        (c_float * len(loss.grad)).from_buffer(loss.grad),  # Use upstream gradient
                        (c_float * B).from_buffer(loss_data),               # Losses can be reused
                        (c_float * len(self.grad)).from_buffer(self.grad),  # Output gradient
                        None,  # probs_out not needed in backward
                        B,
                        C
                    )

            loss._backward = _backward
            loss._prev = [self, target]

        return loss.mean()

    def relu(self):
        out_data = get_buffer(len(self.data))

        lib.tensor_relu(
            (c_float * len(self.data)).from_buffer(self.data),
            (c_float * len(out_data)).from_buffer(out_data),
            len(self.data)
        )
        
        out = Tensor(out_data, requires_grad=self.requires_grad, shape=self.shape)

        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                
                if self.requires_grad:
                    lib.tensor_relu_backward(
                        (c_float * len(self.data)).from_buffer(out.grad),
                        (c_float * len(self.data)).from_buffer(self.data),
                        (c_float * len(self.data)).from_buffer(self.grad),
                        len(self.data)
                    )

            out._backward = _backward
            out._prev = [self]

        return out
    
    def mean(self):
        result = lib.tensor_mean((c_float * len(self.data)).from_buffer(self.data), len(self.data))
        out = Tensor([result], requires_grad=self.requires_grad)
        
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    grad_val = out.grad[0] / len(self.data)

                    grad_array = array.array('f', [grad_val] * len(self.grad))
                    
                    lib.tensor_add_inplace(
                        (c_float * len(self.grad)).from_buffer(self.grad),
                        (c_float * len(grad_array)).from_buffer(grad_array),
                        len(self.grad)
                    )

            out._backward = _backward
            out._prev = [self]
        
        return out

    def sum(self):
        result = lib.tensor_sum((c_float * len(self.data)).from_buffer(self.data), len(self.data))
        out = Tensor([result], requires_grad=self.requires_grad)

        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    grad_val = out.grad[0]

                    grad_array = array.array('f', [grad_val] * len(self.grad))

                    lib.tensor_add_inplace(
                        (c_float * len(self.grad)).from_buffer(self.grad),
                        (c_float * len(grad_array)).from_buffer(grad_array),
                        len(self.grad)
                    )

            out._backward = _backward
            out._prev = [self]

        return out

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={list(self.data)}, grad={list(self.grad) if self.grad else None})"
    
def _init_backend():
    result = lib.tensor_ops_init()
    if result != 0:
        raise RuntimeError("tensor_ops_init failed (AVX2 unsupported?)")
    
_init_backend()