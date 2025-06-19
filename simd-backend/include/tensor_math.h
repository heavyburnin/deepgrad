// tensor_math.h

#ifndef TENSOR_MATH_H
#define TENSOR_MATH_H

#include <stddef.h>  // for size_t

// Element-wise math operations
void tensor_add(const float* a, const float* b, float* out, size_t n, size_t batch_size);
void tensor_sub(const float* a, const float* b, float* out, size_t n, size_t batch_size);
void tensor_mul(const float* a, const float* b, float* out, size_t n, size_t batch_size);
void tensor_div(const float* a, const float* b, float* out, size_t n, size_t batch_size);

// Gradient operations
void tensor_add_grad(const float* dout, const float* a, const float* b, float* da, float* db, size_t n, size_t batch_size);
void tensor_sub_grad(const float* dout, const float* a, const float* b, float* da, float* db, size_t n, size_t batch_size);
void tensor_mul_grad(const float* dout, const float* a, const float* b, float* da, float* db, size_t n, size_t batch_size);
void tensor_div_grad(const float* dout, const float* a, const float* b, float* da, float* db, size_t n, size_t batch_size);

#endif // TENSOR_MATH_H