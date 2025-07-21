// tensor_conv2d.h - Header for Conv2D GEMM implementation
#ifndef TENSOR_CONV2D_H
#define TENSOR_CONV2D_H

#include <stddef.h>

// Forward pass
void conv2d_forward_gemm(const float* restrict input,
                         const float* restrict weight,
                         const float* restrict bias,
                         float* restrict output,
                         size_t N, size_t C_in, size_t H_in, size_t W_in,
                         size_t C_out, size_t K_h, size_t K_w,
                         size_t stride_h, size_t stride_w,
                         size_t pad_h, size_t pad_w);

// Backward pass
void conv2d_backward_gemm(const float* restrict input,
                          const float* restrict weight,
                          const float* restrict grad_out,
                          float* restrict grad_input,
                          float* restrict grad_weight,
                          float* restrict grad_bias,
                          size_t N, size_t C_in, size_t H_in, size_t W_in,
                          size_t C_out, size_t K_h, size_t K_w,
                          size_t stride_h, size_t stride_w,
                          size_t pad_h, size_t pad_w);

#endif // TENSOR_CONV2D_H
