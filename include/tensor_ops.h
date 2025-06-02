// tensor_ops.h

#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Core element-wise operations (SIMD enabled)
void tensor_add(const float*, const float*, float*, size_t);
void tensor_sub(const float*, const float*, float*, size_t);
void tensor_mul(const float*, const float*, float*, size_t);
void tensor_div(const float*, const float*, float*, size_t);
void tensor_exp(const float*, float*, size_t);
void tensor_relu(const float*, float*, size_t);

// Reductions and activations
void tensor_softmax_cross_entropy_with_probs(const float* logits,
                                             const float* labels,
                                             float* loss_out,
                                             float* probs_out,
                                             size_t class_count);
void tensor_matmul(const float* A, const float* B, float* C,
                   size_t M, size_t K, size_t N);
void tensor_matmul_batch_transposedB(const float* A, const float* B_T, float* C,
                                     size_t batch, size_t M, size_t K, size_t N);
float tensor_sum(const float*, size_t);
float tensor_mean(const float*, size_t);


void tensor_matmul(const float* A, const float* B, float* C,
                   size_t M, size_t K, size_t N);

// JIT Batching for runtime batch-aware dispatch
void tensor_op_jit_2in(void (*op)(const float*, const float*, float*, size_t),
                       const float* a, const float* b, float* out,
                       size_t batch, size_t stride);

void tensor_op_jit_1in(void (*op)(const float*, float*, size_t),
                       const float* a, float* out,
                       size_t batch, size_t stride);

void tensor_matmul_batch_jit(const float* A, const float* B, float* C,
                             size_t batch, size_t M, size_t N, size_t K);

void tensor_op_jit_softmax_ce_with_probs(const float* logits,
                                         const float* labels,
                                         float* losses,
                                         float* probs_out,
                                         size_t batch,
                                         size_t class_count);
                                          
void tensor_matmul_batch(const float* A, const float* B, float* C,
                         size_t batch, size_t M, size_t K, size_t N);

void tensor_matmul_batch_transposedB(const float* A, const float* B_T, float* C,
                                     size_t batch, size_t M, size_t K, size_t N);

#ifdef __cplusplus
}
#endif