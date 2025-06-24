#ifndef TENSOR_POOL2D_H
#define TENSOR_POOL2D_H

#include <stddef.h>

// Forward pass off avgpool2d
void avgpool2d_forward(const float* input, float* output,
                       size_t N, size_t C, size_t H, size_t W,
                       size_t kernel_h, size_t kernel_w,
                       size_t stride_h, size_t stride_w);

// Backward pass of avgpool2d
void avgpool2d_backward(const float* grad_out, float* grad_input,
                        size_t N, size_t C, size_t H, size_t W,
                        size_t kernel_h, size_t kernel_w,
                        size_t stride_h, size_t stride_w);

// Forward pass off maxpool2d         
void maxpool2d_forward(const float*, float*, size_t, size_t, size_t, size_t,
                       size_t, size_t, size_t, size_t);

// Forward pass off maxpool2d
void maxpool2d_backward(const float*, const float*, float*, size_t, size_t, size_t, size_t,
                        size_t, size_t, size_t, size_t);

#endif // TENSOR_POOL2D_H
