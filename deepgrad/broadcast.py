from deepgrad.backend import SimdTensorBackend
from functools import lru_cache
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
def compute_broadcast_shape(shape1, shape2):
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

from deepgrad.backend import SimdTensorBackend
from ctypes import cast, c_float, c_int, c_size_t, POINTER
c_float_p = POINTER(c_float)

def broadcast_to_shape(data, from_shape, to_shape, from_size):
    # Validate shapes for broadcasting
    ndim_from = len(from_shape)
    ndim_to = len(to_shape)
    
    # Pad from_shape with leading 1's to match to_shape's rank
    from_shape_padded = (1,) * (ndim_to - ndim_from) + from_shape
    
    # Check rank compatibility
    if len(from_shape_padded) != ndim_to:
        raise ValueError(f"Cannot broadcast shape {from_shape} to {to_shape}: rank mismatch")
    
    # Check dimension compatibility
    for orig_dim, target_dim in zip(from_shape_padded, to_shape):
        if orig_dim != 1 and orig_dim != target_dim:
            raise ValueError(f"Cannot broadcast shape {from_shape} to {to_shape}: dimension {orig_dim} incompatible with {target_dim}")

    # Return original data if shapes match
    if from_shape == to_shape:
        return data

    # Compute output size
    out_size = 1
    for dim in to_shape:
        out_size *= dim

    # Prepare output buffer
    out = (c_float * out_size)()

    # Convert shapes to ctypes arrays
    from_shape_arr = (c_int * ndim_from)(*from_shape) if ndim_from > 0 else None
    to_shape_arr = (c_int * ndim_to)(*to_shape)

    # If scalar input, fill the buffer directly
    if ndim_from == 0 or from_size == 1:
        val = data[0]
        for i in range(out_size):
            out[i] = val
        return out

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
