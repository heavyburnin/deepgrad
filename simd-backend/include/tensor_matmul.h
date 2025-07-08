#ifndef TENSOR_MATMUL_H
#define TENSOR_MATMUL_H

#pragma once

#include <stddef.h>
#include <stdbool.h>

typedef enum {
    MATMUL_FORWARD,
    MATMUL_BACKWARD
} PassMode;

// Unified entry point for both forward and backward
void tensor_matmul(
    PassMode mode,
    const float* A, const float* B, const float* grad_out,
    float* C_or_gradA, float* grad_B,
    size_t batch, size_t M, size_t K, size_t N,
    bool accumulate
);

// Forward pass: C = A @ B
void matmul_forward(
    const float* A, const float* B,
    float* C,
    size_t batch, size_t M, size_t K, size_t N
);

// Backward pass: computes grad_A and/or grad_B
void matmul_backward(
    const float* A, const float* B, const float* grad_out,
    float* grad_A, float* grad_B,
    size_t batch, size_t M, size_t K, size_t N,
    bool accumulate
);

// Releases internal transpose buffers
void tensor_matmul_free_cache(void);

#endif // TENSOR_MATMUL_H
