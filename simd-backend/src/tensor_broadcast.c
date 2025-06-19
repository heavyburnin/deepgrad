// tensor_broadcast.c

#include "tensor_broadcast.h"
#include <immintrin.h>   // For AVX2 intrinsics (__m256, etc.)
#include <stddef.h>      // For size_t
#include <stdio.h>       // For fprintf
#include <stdlib.h>      // For aligned_alloc, free
#include <string.h>      // For memset
#include <omp.h>         // For OpenMP

// Efficient broadcasting of a single row [1, N] → [B, N]
void tensor_broadcast_row(const float* input, float* output, size_t B, size_t N) {
    if (!input || !output) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_broadcast_row\n");
        return;
    }

    #pragma omp parallel for if (B > 4)
    for (size_t b = 0; b < B; ++b) {
        // Use AVX2 vectorized copy for larger chunks
        size_t i = 0;
        for (; i + 8 <= N; i += 8) {
            __m256 vec = _mm256_loadu_ps(&input[i]);
            _mm256_storeu_ps(&output[b * N + i], vec);
        }
        
        // Handle remaining elements
        for (; i < N; ++i) {
            output[b * N + i] = input[i];
        }
    }
}

void tensor_broadcast_col(const float* input, float* output, size_t B, size_t N) {
    if (!input || !output) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_broadcast_col\n");
        return;
    }

    #pragma omp parallel for if(B > 4)
    for (size_t i = 0; i < B; ++i) {
        float val = input[i];
        __m256 val_vec = _mm256_set1_ps(val);
        
        // Process 8 elements at a time with AVX2
        size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            _mm256_storeu_ps(&output[i * N + j], val_vec);
        }
        
        // Handle remaining elements
        for (; j < N; ++j) {
            output[i * N + j] = val;
        }
    }
}

// tensor_unbroadcast_sum_axes: generalized reduction over broadcasted axes
void tensor_unbroadcast_sum_axes(
    const float* grad,
    float* out,
    const size_t* shape_out,
    const size_t* strides_grad,
    const size_t* strides_out,
    size_t ndim,
    size_t total_grad,
    size_t total_out
) {
    // Initialize output to zero using vectorized operations
    #pragma omp parallel
    {
        const size_t vec_size = 8; // AVX2 processes 8 floats at once
        __m256 zero_vec = _mm256_setzero_ps();
        
        #pragma omp for
        for (size_t i = 0; i < total_out - (total_out % vec_size); i += vec_size) {
            _mm256_storeu_ps(&out[i], zero_vec);
        }
        
        // Handle remaining elements
        #pragma omp for
        for (size_t i = total_out - (total_out % vec_size); i < total_out; ++i) {
            out[i] = 0.0f;
        }
    }

    // Create thread-local buffers to avoid atomic operations
    #pragma omp parallel
    {
        float* local_out = (float*)aligned_alloc(32, total_out * sizeof(float));
        memset(local_out, 0, total_out * sizeof(float));
        
        #pragma omp for
        for (size_t i = 0; i < total_grad; ++i) {
            size_t idx = i;
            size_t out_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                size_t pos = idx / strides_grad[d];
                idx %= strides_grad[d];
                if (shape_out[d] == 1)
                    continue;  // broadcasted, skip
                out_idx += pos * strides_out[d];
            }
            local_out[out_idx] += grad[i];
        }
        
        // Reduce thread-local results to the output array
        #pragma omp critical
        {
            const size_t vec_size = 8;
            for (size_t i = 0; i < total_out - (total_out % vec_size); i += vec_size) {
                __m256 out_vec = _mm256_loadu_ps(&out[i]);
                __m256 local_vec = _mm256_loadu_ps(&local_out[i]);
                _mm256_storeu_ps(&out[i], _mm256_add_ps(out_vec, local_vec));
            }
            
            // Handle remaining elements
            for (size_t i = total_out - (total_out % vec_size); i < total_out; ++i) {
                out[i] += local_out[i];
            }
        }
        
        free(local_out);
    }
}
