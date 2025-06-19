// tensor_utils.c

#include "tensor_utils.h"
#include <immintrin.h>   // For AVX intrinsics
#include <stdio.h>       // For fprintf, stderr
#include <stdlib.h>      // For NULL
#include <mm_malloc.h>   // For _mm_malloc, _mm_free
#include <math.h>

float hsum256_ps(__m256 v) {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_movehdup_ps(sum128));
    return _mm_cvtss_f32(sum128);
}

__m256 log256_ps(__m256 x) {
    __m256 one = _mm256_set1_ps(1.0f);

    // avoid log(0)
    x = _mm256_max_ps(x, _mm256_set1_ps(1e-30f));

    __m256i ix = _mm256_castps_si256(x);
    __m256i exp = _mm256_srli_epi32(ix, 23);

    __m256 e = _mm256_cvtepi32_ps(_mm256_sub_epi32(exp, _mm256_set1_epi32(127)));

    __m256i mant_mask = _mm256_set1_epi32(0x007FFFFF);
    __m256 mantissa = _mm256_or_ps(_mm256_castsi256_ps(_mm256_and_si256(ix, mant_mask)),
                                   _mm256_castsi256_ps(_mm256_set1_epi32(0x3f800000)));

    __m256 m = _mm256_sub_ps(mantissa, one);

    // polynomial approx: log(1 + m)
    __m256 p = _mm256_set1_ps(7.0376836292E-2f);
    p = _mm256_fmadd_ps(p, m, _mm256_set1_ps(-1.1514610310E-1f));
    p = _mm256_fmadd_ps(p, m, _mm256_set1_ps(1.1676998740E-1f));
    p = _mm256_fmadd_ps(p, m, _mm256_set1_ps(-1.2420140846E-1f));
    p = _mm256_fmadd_ps(p, m, _mm256_set1_ps(+1.4249322787E-1f));
    p = _mm256_fmadd_ps(p, m, _mm256_set1_ps(-1.6668057665E-1f));
    p = _mm256_fmadd_ps(p, m, _mm256_set1_ps(+2.0000714765E-1f));
    p = _mm256_fmadd_ps(p, m, _mm256_set1_ps(-2.4999993993E-1f));
    p = _mm256_fmadd_ps(p, m, _mm256_set1_ps(+3.3333331174E-1f));
    p = _mm256_mul_ps(p, m);

    return _mm256_add_ps(_mm256_mul_ps(e, _mm256_set1_ps(0.69314718056f)), p);
}

__m256 exp256_ps(__m256 x) {
    const __m256 ln2 = _mm256_set1_ps(0.69314718056f);
    const __m256 inv_ln2 = _mm256_set1_ps(1.44269504089f);  // 1/ln(2)
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256i bias = _mm256_set1_epi32(127);

    // clamp x to avoid overflow
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.336544f)), _mm256_set1_ps(88.722839f));

    // n = floor(x / ln2 + 0.5)
    __m256 fx = _mm256_fmadd_ps(x, inv_ln2, _mm256_set1_ps(0.5f));
    __m256i emm0 = _mm256_cvttps_epi32(fx);
    fx = _mm256_cvtepi32_ps(emm0);

    // r = x - n * ln2
    __m256 r = _mm256_fnmadd_ps(fx, ln2, x);

    // polynomial approximation for exp(r)
    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(5.0000001201E-1f));
    y = _mm256_fmadd_ps(y, r, one);

    // 2^n
    __m256i pow2n = _mm256_slli_epi32(_mm256_add_epi32(emm0, bias), 23);
    __m256 result = _mm256_mul_ps(y, _mm256_castsi256_ps(pow2n));
    return result;
}

float hmax256_ps(__m256 v) {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 max128 = _mm_max_ps(low, high);
    max128 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    max128 = _mm_max_ss(max128, _mm_movehdup_ps(max128));
    return _mm_cvtss_f32(max128);
}

// Utility to allocate or reuse aligned memory
float* get_cached_buffer(float** buf, size_t* current_size, size_t required_size) {
    if (*current_size < required_size) {
        if (*buf) _mm_free(*buf);
        *buf = (float*)_mm_malloc(required_size * sizeof(float), 64);
        if (!*buf) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            *current_size = 0;
            return NULL;
        }
        *current_size = required_size;
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
