#ifndef TENSOR_MATMUL_H
#define TENSOR_MATMUL_H

#pragma once

#include <stddef.h>
#include <stdbool.h>

typedef enum {
    MATMUL_FORWARD,
    MATMUL_BACKWARD
} PassMode;

void tensor_matmul(
    PassMode mode,
    const float* A, const float* B, const float* grad_out,
    float* C_or_A, float* grad_B,
    size_t batch, size_t M, size_t K, size_t N,
    bool accumulate
);

void matmul_forward(
    const float* A, const float* B,
    float* C,
    size_t batch, size_t M, size_t K, size_t N
);

void matmul_backward(
    const float* A, const float* B, const float* grad_out,
    float* grad_A, float* grad_B,
    size_t batch, size_t M, size_t K, size_t N,
    bool accumulate
);

void tensor_matmul_free_cache();

#endif // TENSOR_MATMUL_H
