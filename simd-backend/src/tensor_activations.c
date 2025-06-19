// tensor_activations.c

#include "tensor_activations.h"
#include <immintrin.h>   // AVX intrinsics: __m256, _mm256_*
#include <stddef.h>      // size_t
#include <stdio.h>       // fprintf
#include <math.h>        // fmaxf
#include <x86intrin.h>   // _mm_prefetch (optional)

void tensor_relu(const float* a, float* out, size_t n) {
    if (!a || !out) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_relu\n");
        return;
    }
    
    size_t i = 0;
    __m256 zero = _mm256_setzero_ps();
    // Process 32 elements at a time
    for (; i + 4*VEC_SIZE <= n; i += 4*VEC_SIZE) {
        _mm_prefetch(a + i + 4*VEC_SIZE, _MM_HINT_T0);
        
        _mm256_storeu_ps(out + i,             _mm256_max_ps(zero, _mm256_loadu_ps(a + i)));
        _mm256_storeu_ps(out + i + VEC_SIZE,   _mm256_max_ps(zero, _mm256_loadu_ps(a + i + VEC_SIZE)));
        _mm256_storeu_ps(out + i + 2*VEC_SIZE, _mm256_max_ps(zero, _mm256_loadu_ps(a + i + 2*VEC_SIZE)));
        _mm256_storeu_ps(out + i + 3*VEC_SIZE, _mm256_max_ps(zero, _mm256_loadu_ps(a + i + 3*VEC_SIZE)));
    }
    for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vmax = _mm256_max_ps(zero, va);
        _mm256_storeu_ps(out + i, vmax);
    }
    for (; i < n; i++) out[i] = fmaxf(0.0f, a[i]);
}

void tensor_relu_backward(const float* grad_output, const float* input, float* grad_input, size_t n) {
    if (!grad_output || !input || !grad_input) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_relu_backward\n");
        return;
    }
    
    size_t i = 0;
    __m256 zero = _mm256_setzero_ps();
    
    // Process 32 elements at a time (4 * 8 = 32 floats)
    for (; i + 4*VEC_SIZE <= n; i += 4*VEC_SIZE) {
        _mm_prefetch(input + i + 4*VEC_SIZE, _MM_HINT_T0);
        _mm_prefetch(grad_output + i + 4*VEC_SIZE, _MM_HINT_T0);
        
        for (size_t j = 0; j < 4; j++) {
            size_t idx = i + j*VEC_SIZE;
            __m256 vinput = _mm256_loadu_ps(input + idx);
            __m256 vgrad = _mm256_loadu_ps(grad_output + idx);
            __m256 vmask = _mm256_cmp_ps(vinput, zero, _CMP_GT_OQ);
            __m256 vresult = _mm256_and_ps(vgrad, vmask);
            __m256 vprev = _mm256_loadu_ps(grad_input + idx);
            __m256 vsum = _mm256_add_ps(vprev, vresult);
            _mm256_storeu_ps(grad_input + idx, vsum);
        }
    }

    for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
        __m256 vinput = _mm256_loadu_ps(input + i);
        __m256 vgrad = _mm256_loadu_ps(grad_output + i);
        __m256 vmask = _mm256_cmp_ps(vinput, zero, _CMP_GT_OQ);
        __m256 vresult = _mm256_and_ps(vgrad, vmask);
        __m256 vprev = _mm256_loadu_ps(grad_input + i);
        __m256 vsum = _mm256_add_ps(vprev, vresult);
        _mm256_storeu_ps(grad_input + i, vsum);
    }
    
    for (; i < n; i++) {
        grad_input[i] += input[i] > 0.0f ? grad_output[i] : 0.0f;
    }
}