import ctypes
import os

# Set up C library interface
SimdTensorBackend = ctypes.cdll.LoadLibrary(os.path.abspath("simd-backend/build/libsimd_tensor_backend.so"))

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
    'sgd_update_inplace': ([c_float_p, c_float_p, c_size_t, c_float], None),


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

__all__ = [
    "SimdTensorBackend",
    "c_float", "c_size_t", "c_int", "c_bool",
    "c_float_p", "c_size_t_p"
]