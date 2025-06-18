import array
from deepgrad.backend import SimdTensorBackend, c_float

_zero_buffer_pool = {}
_broadcast_cache = {}

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

    return array.array('f', _zero_buffer_pool[size])

def buffer_from(g):
    return (c_float * len(g)).from_buffer(g)

def get_broadcast_cache_key(data, from_shape, to_shape):
    return (id(data), from_shape, to_shape)

def get_broadcasted(key):
    return _broadcast_cache.get(key)

def set_broadcasted(key, value):
    _broadcast_cache[key] = value

def broadcast_to_shape(data, from_shape, to_shape):
    if from_shape == to_shape:
        return data

    rank_diff = len(to_shape) - len(from_shape)
    from_shape = (1,) * rank_diff + from_shape

    # Compute total size
    out_size = 1
    for dim in to_shape:
        out_size *= dim

    # Allocate and fill output buffer
    out = array.array('f', [0.0] * out_size)

    # Index traversal over broadcasted shape
    for out_idx in range(out_size):
        # Compute N-D index
        idx = []
        stride = out_size
        for dim in to_shape:
            stride //= dim
            idx.append((out_idx // stride) % dim)

        # Map to source index
        src_idx = 0
        stride = 1
        for i in reversed(range(len(to_shape))):
            dim = from_shape[i]
            if dim == 1:
                continue  # broadcasted dim
            idx_val = idx[i]
            stride *= dim
            src_idx = src_idx * dim + idx_val

        out[out_idx] = data[src_idx % len(data)]

    return out
