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

def broadcast_to_shape(data, from_shape, to_shape, from_size):
    if from_shape == to_shape:
        return data

    ndim_from = len(from_shape)
    ndim_to = len(to_shape)

    # Compute output size
    out_size = 1
    for dim in to_shape:
        out_size *= dim

    # Prepare output buffer
    out = (c_float * out_size)()

    # Convert input shapes to ctypes arrays
    from_shape_arr = (c_int * ndim_from)(*from_shape)
    to_shape_arr = (c_int * ndim_to)(*to_shape)

    # Call the C function
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
