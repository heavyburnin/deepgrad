/**
 * tensor_ops.h
 * 
 * SIMD-accelerated tensor operations for deep learning
 */

#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <stddef.h>

/**
 * Element-wise addition of two tensors
 * 
 * @param a First input tensor
 * @param b Second input tensor
 * @param out Output tensor
 * @param n Number of elements
 */
void tensor_add(const float* a, const float* b, float* out, size_t n);

/**
 * Element-wise subtraction of two tensors
 * 
 * @param a First input tensor
 * @param b Second input tensor
 * @param out Output tensor
 * @param n Number of elements
 */
void tensor_sub(const float* a, const float* b, float* out, size_t n);

/**
 * Element-wise multiplication of two tensors
 * 
 * @param a First input tensor
 * @param b Second input tensor
 * @param out Output tensor
 * @param n Number of elements
 */
void tensor_mul(const float* a, const float* b, float* out, size_t n);

/**
 * Element-wise division of two tensors
 * 
 * @param a First input tensor
 * @param b Second input tensor
 * @param out Output tensor
 * @param n Number of elements
 */
void tensor_div(const float* a, const float* b, float* out, size_t n);

/**
 * Element-wise exponential function
 * 
 * @param a Input tensor
 * @param out Output tensor
 * @param n Number of elements
 */
void tensor_exp(const float* a, float* out, size_t n);

/**
 * Element-wise ReLU activation function
 * 
 * @param a Input tensor
 * @param out Output tensor
 * @param n Number of elements
 */
void tensor_relu(const float* a, float* out, size_t n);

/**
 * Softmax cross entropy with probabilities
 * 
 * @param logits Input logits tensor
 * @param labels Ground truth labels tensor
 * @param loss_out Scalar loss output
 * @param probs_out Output probabilities tensor
 * @param class_count Number of classes
 */
void tensor_softmax_cross_entropy_with_probs(const float* logits,
                                             const float* labels,
                                             float* loss_out,
                                             float* probs_out,
                                             size_t class_count);

/**
 * Matrix multiplication
 * 
 * @param A First input matrix (M x K)
 * @param B Second input matrix (K x N)
 * @param C Output matrix (M x N)
 * @param M Number of rows in A and C
 * @param K Number of columns in A and rows in B
 * @param N Number of columns in B and C
 */
void tensor_matmul(const float* A, const float* B, float* C, 
                   size_t M, size_t K, size_t N);

/**
 * Sum reduction of tensor elements
 * 
 * @param input Input tensor
 * @param len Number of elements
 * @return Sum of all elements
 */
float tensor_sum(const float* input, size_t len);

/**
 * Mean reduction of tensor elements
 * 
 * @param input Input tensor
 * @param len Number of elements
 * @return Mean of all elements
 */
float tensor_mean(const float* input, size_t len);

/**
 * JIT batching wrapper for binary operations
 * 
 * @param op Function pointer to binary operation
 * @param a First input tensor
 * @param b Second input tensor
 * @param out Output tensor
 * @param batch Batch size
 * @param stride Stride size (elements per batch)
 */
void tensor_op_jit_2in(void (*op)(const float*, const float*, float*, size_t),
                       const float* a, const float* b, float* out,
                       size_t batch, size_t stride);

/**
 * JIT batching wrapper for unary operations
 * 
 * @param op Function pointer to unary operation
 * @param a Input tensor
 * @param out Output tensor
 * @param batch Batch size
 * @param stride Stride size (elements per batch)
 */
void tensor_op_jit_1in(void (*op)(const float*, float*, size_t),
                       const float* a, float* out,
                       size_t batch, size_t stride);

/**
 * JIT batching wrapper for softmax cross entropy with probabilities
 * 
 * @param logits Input logits tensor
 * @param labels Ground truth labels tensor
 * @param losses Output losses tensor
 * @param probs_out Output probabilities tensor
 * @param batch Batch size
 * @param class_count Number of classes
 */
void tensor_op_jit_softmax_ce_with_probs(const float* logits,
                                         const float* labels,
                                         float* losses,
                                         float* probs_out,
                                         size_t batch,
                                         size_t class_count);

/**
 * JIT batching wrapper for matrix multiplication
 * 
 * @param A First input tensor (batch x M x K)
 * @param B Second input matrix (K x N)
 * @param C Output tensor (batch x M x N)
 * @param batch Batch size
 * @param M Number of rows in each A matrix and C matrix
 * @param K Number of columns in each A matrix and rows in B
 * @param N Number of columns in B and each C matrix
 */
void tensor_matmul_batch_jit(const float* A, const float* B, float* C,
                             size_t batch, size_t M, size_t K, size_t N);

#endif /* TENSOR_OPS_H */