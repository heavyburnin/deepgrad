// tensor_utils.c

#include "tensor_utils.h"
#include <immintrin.h>   // For AVX intrinsics
#include <stdio.h>       // For fprintf, stderr
#include <stdlib.h>      // For NULL
#include <mm_malloc.h>   // For _mm_malloc, _mm_free
#include <math.h>

// Utility to allocate or reuse aligned memory
float* get_cached_buffer(float** buf, size_t* current_size, size_t required_size) {
    // Round up to next multiple of 8 for safe AVX loads
    size_t padded_size = ((required_size + 7) / 8) * 8;

    if (*current_size < padded_size) {
        if (*buf) _mm_free(*buf);
        *buf = (float*)_mm_malloc(padded_size * sizeof(float), 64);
        if (!*buf) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            *current_size = 0;
            return NULL;
        }
        *current_size = padded_size;
    }
    return *buf;
}

void tensor_add_inplace(float* target, const float* source, size_t size) {
    if (!target || !source) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_add_inplace\n");
        return;
    }

    size_t i = 0;
    const size_t simd_width = 8; // AVX2 processes 8 floats at once
    size_t simd_end = size - (size % simd_width);

    // SIMD loop
    #pragma omp parallel for
    for (i = 0; i < simd_end; i += simd_width) {
        __m256 t = _mm256_loadu_ps(&target[i]);
        __m256 s = _mm256_loadu_ps(&source[i]);
        __m256 result = _mm256_add_ps(t, s);
        _mm256_storeu_ps(&target[i], result);
    }

    // Handle tail with scalar addition
    for (i = simd_end; i < size; i++) {
        target[i] += source[i];
    }
}

void tensor_fill_inplace(float* data, float value, size_t size) {
    size_t i = 0;

    // Vectorize using AVX2 intrinsics for float
    __m256 val_vec = _mm256_set1_ps(value);

    // Process in chunks of 8 floats (256 bits)
    size_t vec_end = size / 8 * 8;

    #pragma omp parallel for
    for (i = 0; i < vec_end; i += 8) {
        _mm256_storeu_ps(&data[i], val_vec);
    }

    // Process any leftover elements
    for (i = vec_end; i < size; i++) {
        data[i] = value;
    }
}

// In-place SGD update with SIMD acceleration
void sgd_update_inplace(float* weights, const float* grads, size_t size, float lr) {
    size_t i = 0;
    size_t vec_size = 8; // AVX2 processes 8 floats
    __m256 lr_vec = _mm256_set1_ps(lr);
    
    // Process 8 elements at a time using AVX2
    for (; i + vec_size <= size; i += vec_size) {
        __m256 w = _mm256_loadu_ps(&weights[i]);
        __m256 g = _mm256_loadu_ps(&grads[i]);
        
        // w = w - lr * g
        __m256 update = _mm256_mul_ps(lr_vec, g);
        w = _mm256_sub_ps(w, update);
        
        _mm256_storeu_ps(&weights[i], w);
    }
    
    // Handle remaining elements
    for (; i < size; i++) {
        weights[i] = weights[i] - lr * grads[i];
    }
}

// In-place Adam update
void adam_update_inplace(
    float* param,      // parameter data
    float* grad,       // gradient
    float* m,          // first moment
    float* v,          // second moment
    size_t size,       // number of elements
    float lr,          // learning rate
    float beta1,       // beta1
    float beta2,       // beta2
    float eps,         // epsilon
    int t              // current timestep
) {
    float beta1_t = 1.0f - powf(beta1, t);
    float beta2_t = 1.0f - powf(beta2, t);

    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        float g = grad[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

        float m_hat = m[i] / beta1_t;
        float v_hat = v[i] / beta2_t;

        param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

void adamw_update_inplace(
    float* param,
    float* grad,
    float* m,
    float* v,
    int size,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int t,
    float weight_decay
) {
    for (int i = 0; i < size; i++) {
        // Decoupled weight decay
        param[i] -= lr * weight_decay * param[i];

        m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];

        float m_hat = m[i] / (1 - powf(beta1, t));
        float v_hat = v[i] / (1 - powf(beta2, t));

        param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

void zero_float_array(float *data, size_t size) {
    size_t i = 0;
    __m256 zero = _mm256_setzero_ps();  // 8 floats = 256 bits

    // Zero in chunks of 8 floats
    for (; i + 8 <= size; i += 8) {
        _mm256_storeu_ps(&data[i], zero);
    }

    // Handle the remaining tail (if not a multiple of 8)
    for (; i < size; ++i) {
        data[i] = 0.0f;
    }
}

// Sanitize gradients by zeroing out non-finite values (NaN, Inf)
void sanitize_gradients(float* data, size_t size) {
    size_t i = 0;
    size_t vec_size = 8; // AVX2 processes 8 floats
    // Process 8 elements at a time using AVX2
    for (; i + vec_size <= size; i += vec_size) {
        __m256 values = _mm256_loadu_ps(&data[i]);
        
        // Create mask for finite values (neither NaN nor Inf)
        __m256 is_finite = _mm256_cmp_ps(values, values, _CMP_EQ_OQ); // NaN check
        __m256 abs_values = _mm256_and_ps(values, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
        __m256 is_not_inf = _mm256_cmp_ps(abs_values, _mm256_set1_ps(INFINITY), _CMP_NEQ_OQ);
        __m256 mask = _mm256_and_ps(is_finite, is_not_inf);
        
        // Zero out non-finite values
        values = _mm256_and_ps(values, mask);
        _mm256_storeu_ps(&data[i], values);
    }
    
    // Handle remaining elements
    for (; i < size; i++) {
        if (!isfinite(data[i])) {
            data[i] = 0.0f;
        }
    }
}
