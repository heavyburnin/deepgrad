from deepgrad.backend import SimdTensorBackend
from operator import mul
from functools import lru_cache, reduce
from typing import Tuple
from ctypes import cast, c_float, c_int, c_size_t, POINTER
c_float_p = POINTER(c_float)

_broadcast_cache = {}

def get_broadcast_cache_key(data, from_shape, to_shape):
    return (id(data), from_shape, to_shape)

def get_broadcasted(key):
    return _broadcast_cache.get(key)

def set_broadcasted(key, value):
    _broadcast_cache[key] = value

@lru_cache(maxsize=128)
def compute_broadcast_shape(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Computes the broadcasted shape for two tensors using NumPy-style broadcasting rules.
    Shapes are aligned from the right, and dimensions are compatible if they are equal or one is 1.
    """
    # Determine the maximum number of dimensions
    max_len = max(len(shape1), len(shape2))
    
    # Pad shapes with ones on the left to match max_len
    s1 = (1,) * (max_len - len(shape1)) + shape1
    s2 = (1,) * (max_len - len(shape2)) + shape2
    
    out_shape = []
    for d1, d2 in zip(s1, s2):
        if d1 == d2 or d1 == 1 or d2 == 1:
            out_shape.append(max(d1, d2))
        else:
            raise ValueError(f"Incompatible shapes for broadcasting: {shape1} and {shape2}")
    
    return tuple(out_shape)

c_float_p = POINTER(c_float)

def broadcast_to_shape(data, from_shape, to_shape, from_size):
    """
    Broadcasts data from from_shape to to_shape using NumPy-style broadcasting.
    """    
    ndim_from = len(from_shape)
    ndim_to = len(to_shape)
    
    # Pad from_shape with leading 1's to match to_shape's rank
    from_shape_padded = (1,) * (ndim_to - ndim_from) + from_shape
    
    # Validate dimension compatibility
    for orig_dim, target_dim in zip(from_shape_padded, to_shape):
        if orig_dim != 1 and orig_dim != target_dim:
            raise ValueError(f"Cannot broadcast shape {from_shape} to {to_shape}: dimension {orig_dim} incompatible with {target_dim}")

    # Return original data if shapes match
    if from_shape == to_shape:
        return data

    # Compute output size
    out_size = reduce(mul, to_shape, 1)
    out = (c_float * out_size)()

    # Handle scalar input
    if ndim_from == 0 or from_size == 1:
        val = data[0]
        for i in range(out_size):
            out[i] = val
        return out

    # Convert shapes to ctypes arrays
    from_shape_arr = (c_int * ndim_from)(*from_shape) if ndim_from > 0 else None
    to_shape_arr = (c_int * ndim_to)(*to_shape)

    # Call backend function
    SimdTensorBackend.broadcast_to_shape(
        cast(data, c_float_p),
        from_shape_arr,
        to_shape_arr,
        c_size_t(ndim_from),
        c_size_t(ndim_to),
        c_size_t(from_size),
        out
    )
    return out

def broadcast_data(data, from_shape, to_shape):
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

def unbroadcast_grad(grad, grad_shape, target_shape, out_grad_buf=None):
    ndim = len(grad_shape)

    if len(target_shape) != ndim:
        target_shape = (1,) * (ndim - len(target_shape)) + target_shape

    grad_sz = 1
    for dim in grad_shape:
        grad_sz *= dim

    out_sz = 1
    for dim in target_shape:
        out_sz *= dim

    target = out_grad_buf if out_grad_buf is not None else (c_float * out_sz)()

    strides_grad = [1] * ndim
    for i in reversed(range(ndim - 1)):
        strides_grad[i] = strides_grad[i + 1] * grad_shape[i + 1]

    strides_out = [1] * ndim
    for i in reversed(range(ndim - 1)):
        strides_out[i] = strides_out[i + 1] * target_shape[i + 1]

    c_shape_out = (c_size_t * ndim)(*target_shape)
    c_strides_grad = (c_size_t * ndim)(*strides_grad)
    c_strides_out = (c_size_t * ndim)(*strides_out)

    SimdTensorBackend.tensor_unbroadcast_sum_axes(
        grad,
        target,
        c_shape_out,
        c_strides_grad,
        c_strides_out,
        ndim,
        grad_sz,
        out_sz,
        bool(out_grad_buf is not None)
    )

    if out_grad_buf is None:
        return target
