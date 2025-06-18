#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <stddef.h>
#include <stdbool.h>

typedef enum {
    MATMUL_FORWARD = 0,
    MATMUL_BACKWARD = 1
} PassMode;

int tensor_ops_init(void);

// Elementwise Operations
void tensor_add(const float* a, const float* b, float* out, size_t n, size_t batch_size);
void tensor_add_grad(const float* dout, const float* a, const float* b, float* da, float* db, size_t n, size_t batch_size);

void tensor_sub(const float* a, const float* b, float* out, size_t n, size_t batch_size);
void tensor_sub_grad(const float* dout, const float* a, const float* b, float* da, float* db, size_t n, size_t batch_size);

void tensor_mul(const float* a, const float* b, float* out, size_t n, size_t batch_size);
void tensor_mul_grad(const float* dout, const float* a, const float* b, float* da, float* db, size_t n, size_t batch_size);

void tensor_div(const float* a, const float* b, float* out, size_t n, size_t batch_size);
void tensor_div_grad(const float* dout, const float* a, const float* b, float* da, float* db, size_t n, size_t batch_size);

// ReLU
void tensor_relu(const float* a, float* out, size_t n);
void tensor_relu_backward(const float* grad_output, const float* input, float* grad_input, size_t n);

// Softmax + Cross Entropy
void tensor_softmax_ce(const float* logits, const float* labels, const float* grad_loss, float* losses,
    float* grad_input, float* probs_out, size_t batch, size_t class_count);

// Matmul
void tensor_matmul( PassMode mode, const float* A, const float* B, const float* grad_out, float* C_or_grad_A, 
    float* grad_B, size_t batch, size_t M, size_t K, size_t N, bool accumulate);

// Broadcasting
void tensor_broadcast_row(const float* input, float* output, size_t B, size_t N);
void tensor_broadcast_col(const float* input, float* output, size_t B, size_t N);
void tensor_unbroadcast_sum_axes(const float* grad, float* out, const size_t* shape_out, const size_t* strides_output,
    const size_t* strides_input, size_t ndim, size_t total_grad, size_t total_out);

// Fill/Add/Update
void tensor_add_inplace(float* target, const float* source, size_t size);
void tensor_fill_inplace(float* data, float value, size_t size);
void sgd_update_inplace(float* weights, const float* grads, size_t size, float lr);

// Zero Gradients
void zero_float_array(float *data, size_t size);

// Gradient Utilities
void sanitize_gradients(float* data, size_t size);

// Reductions
float tensor_sum(const float* input, size_t len);
float tensor_mean(const float* input, size_t len);

#endif // TENSOR_OPS_H