#ifndef TENSOR_CONV_GEMM_H
#define TENSOR_CONV_GEMM_H

#include <stddef.h>
#include <stdbool.h>

// Forward pass using GEMM
void conv2d_forward_gemm(
    const float* input,       // [N, C_in, H_in, W_in]
    const float* weight,      // [C_out, C_in, K_h, K_w]
    const float* bias,        // [C_out] or NULL
    float* output,            // [N, C_out, H_out, W_out]
    size_t N,                 // batch size
    size_t C_in,
    size_t H_in,
    size_t W_in,
    size_t C_out,
    size_t K_h,
    size_t K_w,
    size_t stride_h,
    size_t stride_w,
    size_t pad_h,
    size_t pad_w
);

// Backward pass using GEMM
void conv2d_backward_gemm(
    const float* input,        // [N, C_in, H_in, W_in]
    const float* weight,       // [C_out, C_in, K_h, K_w]
    const float* grad_out,     // [N, C_out, H_out, W_out]
    float* grad_input,         // [N, C_in, H_in, W_in]
    float* grad_weight,        // [C_out, C_in, K_h, K_w]
    float* grad_bias,          // [C_out] or NULL
    size_t N,
    size_t C_in,
    size_t H_in,
    size_t W_in,
    size_t C_out,
    size_t K_h,
    size_t K_w,
    size_t stride_h,
    size_t stride_w,
    size_t pad_h,
    size_t pad_w
);

#endif // TENSOR_CONV_GEMM_H
