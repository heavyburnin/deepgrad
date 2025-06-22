from deepgrad.backend import SimdTensorBackend, c_float, c_float_p, c_int, c_size_t
from ctypes import cast

_broadcast_cache = {}

def get_broadcast_cache_key(data, from_shape, to_shape):
    return (id(data), from_shape, to_shape)

def get_broadcasted(key):
    return _broadcast_cache.get(key)

def set_broadcasted(key, value):
    _broadcast_cache[key] = value

def broadcast_to_shape(data, from_shape, to_shape, from_size):
    if from_shape == to_shape:
        return data

    rank_diff = len(to_shape) - len(from_shape)
    padded_from_shape = (1,) * rank_diff + from_shape

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
