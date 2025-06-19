// tensor_reductions.c

#include "tensor_reductions.h"
#include "tensor_utils.h"   // For hsum256_ps or other SIMD helpers

#include <immintrin.h>      // For AVX intrinsics
#include <stddef.h>         // For 
#include <stdio.h>          // Error logging
#include <omp.h>            // For OpenMP multithreading

float tensor_sum(const float* input, size_t len) {
    if (!input) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_sum\n");
        return 0.0f;
    }

    float total_sum = 0.0f;
    size_t vec_end = len - (len % 8);

    // Use OpenMP with private SIMD accumulators
    #pragma omp parallel reduction(+:total_sum)
    {
        __m256 vsum = _mm256_setzero_ps();

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < vec_end; i += 8) {
            __m256 v = _mm256_loadu_ps(input + i);
            vsum = _mm256_add_ps(vsum, v);
        }

        // Reduce the SIMD register to scalar
        float partial_sum = hsum256_ps(vsum);

        // Add thread-local partial sum to the global reduction
        total_sum += partial_sum;

        // Implicit barrier here due to reduction
    }

    // Handle remaining elements
    for (size_t i = vec_end; i < len; i++) {
        total_sum += input[i];
    }

    return total_sum;
}

float tensor_mean(const float* input, size_t len) {
    if (!input) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_mean\n");
        return 0.0f;
    }
    
    if (len == 0) {
        fprintf(stderr, "Error: Division by zero in tensor_mean\n");
        return 0.0f;
    }
    
    return tensor_sum(input, len) / (float)len;
}
