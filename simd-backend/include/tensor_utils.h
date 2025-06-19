// tensor_utils.h

#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include <immintrin.h>  // For __m256, __m128, intrinsics
#include <stddef.h>     // For size_t

// Horizontal sum of 8-float AVX vector
float hsum256_ps(__m256 v);

// Horizontal max of 8-float AVX vector
float hmax256_ps(__m256 v);

// Natural log (ln) approximation of 8-float AVX vector
__m256 log256_ps(__m256 x);

// Exponential approximation of 8-float AVX vector
__m256 exp256_ps(__m256 x);

// Aligned memory utility
float* get_cached_buffer(float** buf, size_t* current_size, size_t required_size);

// Add/Fill/Update
void tensor_add_inplace(float* target, const float* source, size_t size);
void tensor_fill_inplace(float* data, float value, size_t size);
void sgd_update_inplace(float* weights, const float* grads, size_t size, float lr);

// Zero Gradients
void zero_float_array(float *data, size_t size);

// Gradient Utilities
void sanitize_gradients(float* data, size_t size);

#endif // TENSOR_UTILS_H
