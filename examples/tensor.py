import ctypes
import os
import array
import math
from functools import lru_cache

# Set up C library interface
lib = ctypes.cdll.LoadLibrary(os.path.abspath("../build/libsimd_tensor_backend.so"))

c_float_p = ctypes.POINTER(ctypes.c_float)
c_size_t_p = ctypes.POINTER(ctypes.c_size_t)

c_size_t = ctypes.c_size_t
c_float = ctypes.c_float

lib.sanitize_gradients.argtypes = [c_float_p, c_size_t]
lib.sgd_update_inplace.argtypes = [c_float_p, c_float_p, c_size_t, c_float_p]
lib.tensor_add.argtypes = [c_float_p, c_float_p, c_float_p, c_size_t]
lib.tensor_sub.argtypes = [c_float_p, c_float_p, c_float_p, c_size_t]
lib.tensor_mul.argtypes = [c_float_p, c_float_p, c_float_p, c_size_t]
lib.tensor_div.argtypes = [c_float_p, c_float_p, c_float_p, c_size_t]
lib.tensor_relu.argtypes = [c_float_p, c_float_p, c_size_t]
lib.tensor_matmul_batch.argtypes = [c_float_p, c_float_p, c_float_p, c_size_t, c_size_t, c_size_t, c_size_t]
lib.tensor_softmax_ce_batch.argtypes = [c_float_p, c_float_p, c_float_p, c_float_p, c_size_t, c_size_t]
lib.tensor_exp.argtypes = [c_float_p, c_float_p, c_size_t]
lib.tensor_sum.restype = c_float
lib.tensor_sum.argtypes = [c_float_p, c_size_t]
lib.tensor_mean.restype = c_float
lib.tensor_mean.argtypes = [c_float_p, c_size_t]
lib.tensor_broadcast_row.argtypes = [c_float_p, c_float_p, c_size_t, c_size_t]
lib.tensor_broadcast_col.argtypes = [c_float_p, c_float_p, c_size_t, c_size_t]
lib.tensor_transpose_batch.argtypes = [c_float_p, c_float_p, c_size_t, c_size_t, c_size_t, c_size_t]
lib.tensor_add_grad.argtypes = [c_float_p, c_float_p, c_float_p, c_float_p, c_float_p, c_size_t]
lib.tensor_sub_grad.argtypes = [c_float_p, c_float_p, c_float_p, c_float_p, c_float_p, c_size_t]
lib.tensor_mul_grad.argtypes = [c_float_p, c_float_p, c_float_p, c_float_p, c_float_p, c_size_t]
lib.tensor_div_grad.argtypes = [c_float_p, c_float_p, c_float_p, c_float_p, c_float_p, c_size_t]
lib.tensor_unbroadcast_sum_axes.argtypes = [
    c_float_p,   # grad
    c_float_p,   # out
    c_size_t_p,  # shape_grad
    c_size_t_p,  # shape_out
    c_size_t_p,  # strides_grad
    c_size_t_p,  # strides_out
    c_size_t,    # ndim
    c_size_t,    # total_grad
    c_size_t,    # total_out
]
lib.tensor_zero.argtypes = [c_float_p, c_size_t]
lib.tensor_add_inplace.argtypes = [c_float_p, c_float_p, c_size_t]
lib.tensor_fill_inplace.argtypes = [c_float_p, c_float, c_size_t]
lib.tensor_add_inplace.restype = None
lib.tensor_fill_inplace.restype = None
lib.sanitize_gradients.restype = None
lib.sgd_update_inplace.restype = None
lib.tensor_add.restype = None
lib.tensor_sub.restype = None
lib.tensor_mul.restype = None
lib.tensor_div.restype = None
lib.tensor_relu.restype = None
lib.tensor_matmul_batch.restype = None
lib.tensor_softmax_ce_batch.restype = None
lib.tensor_exp.restype = None
lib.tensor_broadcast_row.restype = None
lib.tensor_broadcast_col.restype = None
lib.tensor_transpose_batch.restype = None
lib.tensor_add_grad.restype = None
lib.tensor_sub_grad.restype = None
lib.tensor_mul_grad.restype = None
lib.tensor_div_grad.restype = None
lib.tensor_unbroadcast_sum_axes.restype = None
lib.tensor_zero.restype = None

lib.tensor_softmax_ce_backward.argtypes = [
    c_float_p,   # grad_loss
    c_float_p,   # probs
    c_float_p,   # target
    c_float_p,   # grad_input
    c_size_t,                  # B
    c_size_t                   # C
]

lib.tensor_softmax_ce_backward.restype = None

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
        self._backward = None
        self._transposed = None  # <- cache slot
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
        if self.requires_grad and self.grad is not None:
            grad_ptr = (ctypes.c_float * len(self.grad)).from_buffer(self.grad)
            lib.tensor_zero(grad_ptr, len(self.grad))
    
    def is_zero_grad(self):
        if self.grad is None:
            return True
        for g in self.grad:
            if g != 0.0:
                return False
        return True

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
        # Return Tensor that shares grad buffer to avoid copies
        if self.grad is None:
            zeros = array.array('f', [0.0] * len(self.data))
            return Tensor(zeros, requires_grad=False, shape=self.shape)
        if isinstance(self.grad, array.array):
            # Wrap the existing grad buffer (no copy)
            return Tensor(self.grad, requires_grad=False, shape=self.shape)
        # fallback: copy if not array.array
        return Tensor(array.array('f', self.grad), requires_grad=False, shape=self.shape)

    def transpose_back(self):
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
            lib.tensor_transpose_batch(
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
            lib.tensor_transpose_batch(
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
                        # 1D case: in-place addition
                        lib.tensor_add_inplace(
                            (ctypes.c_float * len(self.grad)).from_buffer(self.grad),
                            (ctypes.c_float * len(out.grad)).from_buffer(out.grad),
                            len(self.grad)
                        )

                    elif len(self.shape) == 2:
                        # 2D case: transpose and then add in-place
                        M, N = self.shape
                        temp = array.array('f', [0.0] * len(self.grad))
                        lib.tensor_transpose_batch(
                            (ctypes.c_float * len(out.grad)).from_buffer(out.grad),
                            (ctypes.c_float * len(temp)).from_buffer(temp),
                            ctypes.c_size_t(2),
                            ctypes.c_size_t(1),  # B = 1 for 2D
                            ctypes.c_size_t(N),
                            ctypes.c_size_t(M)
                        )
                        lib.tensor_add_inplace(
                            (ctypes.c_float * len(self.grad)).from_buffer(self.grad),
                            (ctypes.c_float * len(temp)).from_buffer(temp),
                            len(self.grad)
                        )

                    elif len(self.shape) == 3:
                        B, M, N = self.shape
                        temp = array.array('f', [0.0] * len(self.grad))
                        lib.tensor_transpose_batch(
                            (ctypes.c_float * len(out.grad)).from_buffer(out.grad),
                            (ctypes.c_float * len(temp)).from_buffer(temp),
                            ctypes.c_size_t(3),
                            ctypes.c_size_t(B),
                            ctypes.c_size_t(N),
                            ctypes.c_size_t(M)
                        )
                        lib.tensor_add_inplace(
                            (ctypes.c_float * len(self.grad)).from_buffer(self.grad),
                            (ctypes.c_float * len(temp)).from_buffer(temp),
                            len(self.grad)
                        )

            out._backward = _backward
            out._prev = [self]

        return out
    
    def transpose(self):
        """Return cached transposed tensor if available."""
        if self._transposed is not None:
            return self._transposed

        if len(self.shape) == 1:
            N = self.shape[0]
            out_shape = (N, 1)
            transposed_data = array.array('f', [0.0] * N)
            for i in range(N):
                transposed_data[i] = self.data[i]

        elif len(self.shape) == 2:
            M, N = self.shape
            out_shape = (N, M)
            transposed_data = array.array('f', [0.0] * (M * N))
            lib.tensor_transpose_batch(
                (ctypes.c_float * len(self.data)).from_buffer(self.data),
                (ctypes.c_float * len(transposed_data)).from_buffer(transposed_data),
                ctypes.c_size_t(2),
                ctypes.c_size_t(1),
                ctypes.c_size_t(M),
                ctypes.c_size_t(N)
            )

        elif len(self.shape) == 3:
            B, M, N = self.shape
            out_shape = (B, N, M)
            transposed_data = array.array('f', [0.0] * (B * M * N))
            lib.tensor_transpose_batch(
                (ctypes.c_float * len(self.data)).from_buffer(self.data),
                (ctypes.c_float * len(transposed_data)).from_buffer(transposed_data),
                ctypes.c_size_t(3),
                ctypes.c_size_t(B),
                ctypes.c_size_t(M),
                ctypes.c_size_t(N)
            )
        else:
            raise NotImplementedError(f"Transpose not implemented for {len(self.shape)}D tensors")

        out = Tensor(transposed_data, requires_grad=self.requires_grad, shape=out_shape)

        # Caching
        self._transposed = out
        out._transposed = self  # inverse cache

        # Preserve autograd logic
        if out.requires_grad:
            def _backward():
                if self.requires_grad and out.grad:
                    if isinstance(self.grad, list):
                        self.grad = array.array('f', self.grad)

                    if len(self.shape) == 1:
                        lib.tensor_add_inplace(
                            (ctypes.c_float * len(self.grad)).from_buffer(self.grad),
                            (ctypes.c_float * len(out.grad)).from_buffer(out.grad),
                            len(self.grad)
                        )

                    elif len(self.shape) == 2:
                        M, N = self.shape
                        temp = array.array('f', [0.0] * len(self.grad))
                        lib.tensor_transpose_batch(
                            (ctypes.c_float * len(out.grad)).from_buffer(out.grad),
                            (ctypes.c_float * len(temp)).from_buffer(temp),
                            ctypes.c_size_t(2),
                            ctypes.c_size_t(1),
                            ctypes.c_size_t(N),
                            ctypes.c_size_t(M)
                        )
                        lib.tensor_add_inplace(
                            (ctypes.c_float * len(self.grad)).from_buffer(self.grad),
                            (ctypes.c_float * len(temp)).from_buffer(temp),
                            len(self.grad)
                        )

                    elif len(self.shape) == 3:
                        B, M, N = self.shape
                        temp = array.array('f', [0.0] * len(self.grad))
                        lib.tensor_transpose_batch(
                            (ctypes.c_float * len(out.grad)).from_buffer(out.grad),
                            (ctypes.c_float * len(temp)).from_buffer(temp),
                            ctypes.c_size_t(3),
                            ctypes.c_size_t(B),
                            ctypes.c_size_t(N),
                            ctypes.c_size_t(M)
                        )
                        lib.tensor_add_inplace(
                            (ctypes.c_float * len(self.grad)).from_buffer(self.grad),
                            (ctypes.c_float * len(temp)).from_buffer(temp),
                            len(self.grad)
                        )

            out._backward = _backward
            out._prev = [self]

        return out

    def backward(self):
        if self.grad is None:
            self.grad = array.array('f', [0.0] * len(self.data))

        # Set initial gradient to 1.0 using C fill
        lib.tensor_fill_inplace(
            (ctypes.c_float * len(self.grad)).from_buffer(self.grad),
            ctypes.c_float(1.0),
            ctypes.c_size_t(len(self.grad))
        )
            
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
            if t._backward is not None:
                t._backward()
            # Sanitize gradients after each backward step
            if t.grad is not None:
                t.sanitize_gradients()

    def __getstate__(self):
        """Prepare object for serialization with dill/pickle"""
        state = self.__dict__.copy()
        
        # Convert array.array to list
        if 'data' in state and isinstance(state['data'], array.array):
            state['data'] = list(state['data'])
        
        # Convert grad array to list if it exists
        if 'grad' in state and state['grad'] is not None:
            if isinstance(state['grad'], array.array):
                state['grad'] = list(state['grad'])
        
        # Remove cached ctypes data
        if '_cdata_cache' in state:
            del state['_cdata_cache']
        
        return state

    def __setstate__(self, state):
        """Restore object after deserialization with dill/pickle"""
        # Convert lists back to array.array
        if 'data' in state and isinstance(state['data'], list):
            state['data'] = array.array('f', state['data'])
        
        if 'grad' in state and state['grad'] is not None:
            if isinstance(state['grad'], list):
                state['grad'] = array.array('f', state['grad'])
        
        self.__dict__.update(state)

    @property
    def _cdata(self):
        """Cached ctypes float* pointer to the tensor's data."""
        if not hasattr(self, '_cdata_cache') or len(self._cdata_cache) != len(self.data):
            self._cdata_cache = (ctypes.c_float * len(self.data)).from_buffer(self.data)
        return self._cdata_cache

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
            
        out_arr = array.array('f', [0.0] * out_sz)

        # Compute strides
        def compute_strides(shape):
            strides = [1] * len(shape)
            for i in reversed(range(len(shape) - 1)):
                strides[i] = strides[i + 1] * shape[i + 1]
            return strides

        strides_grad = compute_strides(grad_shape)
        strides_out = compute_strides(shape)

        # Convert to ctypes arrays
        c_grad = (ctypes.c_float * grad_sz).from_buffer(grad_arr)
        c_out = (ctypes.c_float * out_sz).from_buffer(out_arr)
        c_shape_grad = (ctypes.c_size_t * ndim)(*grad_shape)
        c_shape_out = (ctypes.c_size_t * ndim)(*shape)
        c_strides_grad = (ctypes.c_size_t * ndim)(*strides_grad)
        c_strides_out = (ctypes.c_size_t * ndim)(*strides_out)

        # Call C function
        lib.tensor_unbroadcast_sum_axes(
            c_grad, c_out,
            c_shape_grad, c_shape_out,
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
        a = self._broadcast_data(self.data, self.shape, out_shape)
        b = self._broadcast_data(other.data, other.shape, out_shape)

        # Prepare output buffer
        out_size = math.prod(out_shape)
        out_data = array.array('f', [0.0] * out_size)
        
        # Call C function using from_buffer to avoid copying
        getattr(lib, op_name)(
            (ctypes.c_float * len(a)).from_buffer(a),
            (ctypes.c_float * len(b)).from_buffer(b),
            (ctypes.c_float * len(out_data)).from_buffer(out_data),
            len(out_data)
        )

        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad, shape=out_shape)

        if out.requires_grad:
            def _backward():
                if self.requires_grad and self.grad is None:
                    self.grad = array.array('f', [0.0] * len(self.data))
                if other.requires_grad and other.grad is None:
                    other.grad = array.array('f', [0.0] * len(other.data))
                
                out_grad = out.grad
                a_broadcasted = self._broadcast_data(self.data, self.shape, out_shape)
                b_broadcasted = self._broadcast_data(other.data, other.shape, out_shape)

                if self.requires_grad or other.requires_grad:
                    self_grad = array.array('f', [0.0] * len(a_broadcasted))
                    other_grad = array.array('f', [0.0] * len(b_broadcasted))
                    getattr(lib, grad_fn_name + '_grad')(
                        (ctypes.c_float * len(out_grad)).from_buffer(out_grad),
                        (ctypes.c_float * len(a_broadcasted)).from_buffer(a_broadcasted),
                        (ctypes.c_float * len(b_broadcasted)).from_buffer(b_broadcasted),
                        (ctypes.c_float * len(self_grad)).from_buffer(self_grad),
                        (ctypes.c_float * len(other_grad)).from_buffer(other_grad),
                        len(out_grad)
                    )
                    if self.requires_grad:
                        self_grad = self._unbroadcast_grad(self_grad, self.shape)
                        self.grad = array.array('f', [a + b for a, b in zip(self.grad, self_grad)])
                    if other.requires_grad:
                        other_grad = self._unbroadcast_grad(other_grad, other.shape)
                        other.grad = array.array('f', [a + b for a, b in zip(other.grad, other_grad)])

            out._backward = _backward
            out._prev = [self, other]

        return out

    def __add__(self, other):
        return self._apply_op(other, 'tensor_add', 'tensor_add')

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._apply_op(other, 'tensor_sub', 'tensor_sub')

    def __rsub__(self, other):
        return Tensor(other, requires_grad=False).__sub__(self)

    def __mul__(self, other):
        return self._apply_op(other, 'tensor_mul', 'tensor_mul')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._apply_op(other, 'tensor_div', 'tensor_div')

    def __rtruediv__(self, other):
        return Tensor(other, requires_grad=False).__truediv__(self)

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
                if out.grad is None or all(g == 0.0 for g in out.grad):
                    return
                
                if self.requires_grad:
                    # Make sure grad arrays are array.array('f')
                    if isinstance(self.grad, list):
                        self.grad = array.array('f', self.grad)
                    if self.grad is None:
                        self.grad = array.array('f', [0.0]*len(self.data))
                    if out.grad is None:
                        out.grad = array.array('f', [0.0]*len(self.data))

                    lib.tensor_relu_backward(
                        (ctypes.c_float * len(self.data)).from_buffer(out.grad),
                        (ctypes.c_float * len(self.data)).from_buffer(self.data),
                        (ctypes.c_float * len(self.data)).from_buffer(self.grad),
                        len(self.data)
                    )

            out._backward = _backward
            out._prev = [self]

        return out

    def matmul_backup(self, other):
        assert isinstance(other, Tensor), "Operand must be a Tensor"
        s1, s2 = self.shape, other.shape

        # Case: [B, M] @ [M, N] => [B, N]
        if len(s1) == 2 and len(s2) == 2:
            B, M = s1
            M2, N = s2
            assert M == M2, f"Incompatible matmul shapes {s1} and {s2}"

            out_data = array.array('f', [0.0] * (B * N))

            lib.tensor_matmul_batch(
                self._cdata,
                other._cdata,
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
                        
                        # Use tensor_add_inplace instead of Python loop
                        lib.tensor_add_inplace(
                            (ctypes.c_float * len(self.grad)).from_buffer(self.grad),
                            (ctypes.c_float * len(dA.data)).from_buffer(dA.data),
                            len(self.grad)
                        )
                    
                    if other.requires_grad:
                        if isinstance(other.grad, list):
                            other.grad = array.array('f', other.grad)
                        
                        a_T = self.transpose()
                        dB = a_T.matmul(out.grad_tensor())
                        
                        # Use tensor_add_inplace instead of Python loop
                        lib.tensor_add_inplace(
                            (ctypes.c_float * len(other.grad)).from_buffer(other.grad),
                            (ctypes.c_float * len(dB.data)).from_buffer(dB.data),
                            len(other.grad)
                        )

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
                self._cdata,
                other._cdata,
                (ctypes.c_float * len(out_data)).from_buffer(out_data),
                B1, M, K1, N
            )

            out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad, shape=(B1, M, N))

            if out.requires_grad:
                def _backward():
                    grad_out = out.grad_tensor()

                    if self.requires_grad:
                        if isinstance(self.grad, list):
                            self.grad = array.array('f', self.grad)
                        b_T = other.transpose()
                        dA = grad_out.matmul(b_T)
                        lib.tensor_add_inplace(self._cdata, dA._cdata, len(self.data))

                    if other.requires_grad:
                        if isinstance(other.grad, list):
                            other.grad = array.array('f', other.grad)
                        a_T = self.transpose()
                        dB = a_T.matmul(grad_out)
                        lib.tensor_add_inplace(other._cdata, dB._cdata, len(other.data))

                out._backward = _backward
                out._prev = [self, other]

            return out

        else:
            raise NotImplementedError(f"Unsupported shapes for matmul: {s1} @ {s2}")

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
                self._cdata,
                other._cdata,
                (ctypes.c_float * len(out_data)).from_buffer(out_data),
                B, 1, M, N
            )

            out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad, shape=(B, N))

            if out.requires_grad:
                def _backward():
                    if out.grad is None or all(g == 0.0 for g in out.grad):
                        return
                    
                    if self.requires_grad:
                        if isinstance(self.grad, list):
                            self.grad = array.array('f', self.grad)
                        
                        b_T = other.transpose()
                        dA = out.grad_tensor().matmul(b_T)
                        
                        # Use tensor_add_inplace instead of Python loop
                        lib.tensor_add_inplace(
                            (ctypes.c_float * len(self.grad)).from_buffer(self.grad),
                            (ctypes.c_float * len(dA.data)).from_buffer(dA.data),
                            len(self.grad)
                        )
                    
                    if other.requires_grad:
                        if isinstance(other.grad, list):
                            other.grad = array.array('f', other.grad)
                        
                        a_T = self.transpose()
                        dB = a_T.matmul(out.grad_tensor())
                        
                        # Use tensor_add_inplace instead of Python loop
                        lib.tensor_add_inplace(
                            (ctypes.c_float * len(other.grad)).from_buffer(other.grad),
                            (ctypes.c_float * len(dB.data)).from_buffer(dB.data),
                            len(other.grad)
                        )

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
                self._cdata,
                other._cdata,
                (ctypes.c_float * len(out_data)).from_buffer(out_data),
                B1, M, K1, N
            )

            out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad, shape=(B1, M, N))

            if out.requires_grad:
                # Precompute once
                #grad_out_c = out.grad_tensor()._cdata
                grad_out_c = (ctypes.c_float * len(out.grad)).from_buffer(out.grad)

                a_T = self.transpose() if other.requires_grad else None
                b_T = other.transpose() if self.requires_grad else None

                def _backward():
                    if out.grad is None or all(g == 0.0 for g in out.grad):
                        return
                    
                    if self.requires_grad:
                        lib.tensor_matmul_batch(
                            grad_out_c,
                            b_T._cdata,
                            self._cdata,  # Direct in-place into self.grad memory
                            B, 1, N, M
                        )

                    if other.requires_grad:
                        lib.tensor_matmul_batch(
                            a_T._cdata,
                            grad_out_c,
                            other._cdata,  # Direct in-place into other.grad memory
                            B, M, 1, N
                        )

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

        lib.tensor_softmax_ce_batch(
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
                    lib.tensor_softmax_ce_backward(
                        (ctypes.c_float * len(loss.grad)).from_buffer(loss.grad),
                        (ctypes.c_float * len(probs.data)).from_buffer(probs.data),
                        (ctypes.c_float * len(target.data)).from_buffer(target.data),
                        (ctypes.c_float * len(self.grad)).from_buffer(self.grad),
                        B, C
                    )

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

                    # Create an array filled with grad_val
                    grad_array = array.array('f', [grad_val] * len(self.grad))

                    # Add grad_array to self.grad in-place using your optimized C function
                    lib.tensor_add_inplace(
                        (ctypes.c_float * len(self.grad)).from_buffer(self.grad),
                        (ctypes.c_float * len(grad_array)).from_buffer(grad_array),
                        len(self.grad)
                    )

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

                    # Create an array filled with grad_val
                    grad_array = array.array('f', [grad_val] * len(self.grad))

                    # Add grad_array to self.grad inplace using your C SIMD function
                    lib.tensor_add_inplace(
                        (ctypes.c_float * len(self.grad)).from_buffer(self.grad),
                        (ctypes.c_float * len(grad_array)).from_buffer(grad_array),
                        len(self.grad)
                    )
                    
            out._backward = _backward
            out._prev = [self]
        
        return out

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={list(self.data)}, grad={list(self.grad) if self.grad else None})"