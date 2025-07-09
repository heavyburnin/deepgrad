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
c_int_p = ctypes.POINTER(ctypes.c_int)

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

   'matmul_forward': ([
      c_float_p,   # const float* A
      c_float_p,   # const float* B
      c_float_p,   # float* C (output)
      c_size_t,    # size_t batch
      c_size_t,    # size_t M
      c_size_t,    # size_t K
      c_size_t     # size_t N
   ], None),

   'matmul_backward': ([
      c_float_p,   # const float* A
      c_float_p,   # const float* B
      c_float_p,   # const float* grad_out
      c_float_p,   # float* grad_A
      c_float_p,   # float* grad_B
      c_size_t,    # size_t batch
      c_size_t,    # size_t M
      c_size_t,    # size_t K
      c_size_t,    # size_t N
      c_bool       # bool accumulate
   ], None),

   'tensor_softmax_ce': ([c_float_p, c_int_p, c_float_p, c_float_p,
                          c_float_p, c_float_p, c_size_t, c_size_t,
                          c_float, c_int], None),

   'tensor_sum': ([c_float_p, c_float_p, c_size_t], c_float),
   'tensor_mean': ([c_float_p, c_float_p, c_size_t], c_float),

   'broadcast_to_shape': ([c_float_p, c_int_p, c_int_p, c_size_t, c_size_t, c_size_t, c_float_p], None),

   'tensor_unbroadcast_sum_axes': ([c_float_p, c_float_p, c_size_t_p, c_size_t_p, c_size_t_p,
                                    c_size_t, c_size_t, c_size_t, c_bool], None),

   'tensor_fill_inplace': ([c_float_p, c_float, c_size_t], None),
   'zero_float_array': ([c_float_p, c_size_t], None),

   'conv2d_forward_gemm': ([c_float_p,  # input
                            c_float_p,  # weight
                            c_float_p,  # bias
                            c_float_p,  # output
                            c_size_t,   # N
                            c_size_t,   # C_in
                            c_size_t,   # H_in
                            c_size_t,   # W_in
                            c_size_t,   # C_out
                            c_size_t,   # K_h
                            c_size_t,   # K_w
                            c_size_t,   # stride_h
                            c_size_t,   # stride_w
                            c_size_t,   # pad_h
                            c_size_t,   # pad_w
                            ], None),

   'conv2d_backward_gemm': ([c_float_p,  # input
                             c_float_p,  # weight
                             c_float_p,  # grad_out
                             c_float_p,  # grad_input
                             c_float_p,  # grad_weight
                             c_float_p,  # grad_bias
                             c_size_t,   # N
                             c_size_t,   # C_in
                             c_size_t,   # H_in
                             c_size_t,   # W_in
                             c_size_t,   # C_out
                             c_size_t,   # K_h
                             c_size_t,   # K_w
                             c_size_t,   # stride_h
                             c_size_t,   # stride_w
                             c_size_t,   # pad_h
                             c_size_t,   # pad_w
                             ], None),

   'avgpool2d_forward': ([c_float_p,  # input
                           c_float_p,  # output
                           c_size_t,   # N
                           c_size_t,   # C
                           c_size_t,   # H
                           c_size_t,   # W
                           c_size_t,   # kernel_h
                           c_size_t,   # kernel_w
                           c_size_t,   # stride_h
                           c_size_t],  # stride_w
                           None),

   'avgpool2d_backward': ([c_float_p,  # grad_out
                           c_float_p,  # grad_input
                           c_size_t,   # N
                           c_size_t,   # C
                           c_size_t,   # H
                           c_size_t,   # W
                           c_size_t,   # kernel_h
                           c_size_t,   # kernel_w
                           c_size_t,   # stride_h
                           c_size_t],  # stride_w
                           None),

   'maxpool2d_forward': ([c_float_p, c_float_p, c_size_t, c_size_t, c_size_t, c_size_t,
                        c_size_t, c_size_t, c_size_t, c_size_t], None),

   'maxpool2d_backward': ([c_float_p, c_float_p, c_float_p, c_size_t, c_size_t, c_size_t, c_size_t,
                           c_size_t, c_size_t, c_size_t, c_size_t], None),

   'adam_update_inplace': ([
      c_float_p,  # param
      c_float_p,  # grad
      c_float_p,  # m
      c_float_p,  # v
      ctypes.c_size_t,
      c_float, c_float, c_float, c_float, c_int], None),

   
   'adamw_update_inplace': ([
      c_float_p,  # param
      c_float_p,  # grad
      c_float_p,  # m
      c_float_p,  # v
      ctypes.c_size_t,
      c_float, c_float, c_float, c_float, c_int, c_float], None),

   'batchnorm_forward_f32': ([
         c_float_p,
         c_float_p,  # out
         c_float_p,  # x_hat
         c_float_p,  # gamma
         c_float_p,  # beta
         c_float_p,  # running_mean
         c_float_p,  # running_var
         c_size_t,   # B
         c_size_t,   # C
         c_size_t,   # H
         c_size_t,   # W
         c_float,    # eps
         c_float,    # momentum
         c_bool], None),

   'batchnorm_backward_f32': ([
         c_float_p,  # x_hat
         c_float_p,  # grad_out
         c_float_p,  # grad_in
         c_float_p,  # grad_gamma
         c_float_p,  # grad_beta
         c_float_p,  # gamma
         c_size_t,   # B
         c_size_t,   # C
         c_size_t,   # H
         c_size_t,   # W
         c_float     # eps
      ], None)

}

# Set function signatures
for func_name, (argtypes, restype) in function_signatures.items():
    func = getattr(SimdTensorBackend, func_name)
    func.argtypes = argtypes
    func.restype = restype
