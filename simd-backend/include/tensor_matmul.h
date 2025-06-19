#ifndef TENSOR_MATMUL_H
#define TENSOR_MATMUL_H

#include <stddef.h>
#include <stdbool.h>

// Mode flag: forward or backward
typedef enum {
    MATMUL_FORWARD,
    MATMUL_BACKWARD
} PassMode;

// Main fused matmul: forward or backward
void tensor_matmul(
    PassMode mode,
    const float* A, const float* B, const float* grad_out,
    float* C_or_A, float* grad_B,
    size_t batch, size_t M, size_t K, size_t N,
    bool accumulate
);

// Free any internal buffers (e.g., cached transposes)
void tensor_matmul_free_cache(void);

#endif // TENSOR_MATMUL_H
