#ifndef TENSOR_MATMUL_H
#define TENSOR_MATMUL_H

#pragma once

#include <stddef.h>
#include <stdbool.h>


void gemm_avx2_fma(const float* A, const float* B, float* C,
                   size_t M, size_t N, size_t K, bool accumulate);

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

#endif // TENSOR_MATMUL_H
