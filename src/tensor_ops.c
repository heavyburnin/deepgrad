// tensor_ops.c

#include "../include/tensor_ops.h"
#include <immintrin.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <cpuid.h>
#include <omp.h>
#include <stdbool.h>

#define VEC_SIZE 8
#define MAX_CLASSES 512

// Check for AVX2 support at runtime
static int has_avx2() {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(7, &eax, &ebx, &ecx, &edx)) {
        return (ebx & bit_AVX2) != 0;
    }
    return 0;
}

// Initialize tensor operations library
int tensor_ops_init() {
    //#if (!has_avx2()) {
    //    fprintf(stderr, "Error: AVX2 not supported on this CPU\n");
    //    return -1;
    //}
     // Hardcode OpenMP thread count
    omp_set_num_threads(8);  // Set to however many threads you want to use
    return 0;
}

// AVX2 horizontal sum helper
inline float hsum_avx(__m256 v) {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

// Improved log approximation for AVX
static inline __m256 log256_ps(__m256 x) {
    // Handle invalid inputs
    __m256 valid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_GT_OQ);
    x = _mm256_max_ps(x, _mm256_set1_ps(FLT_MIN)); // Avoid log(0)
    
    __m256 one = _mm256_set1_ps(1.0f);
    
    // Extract exponent
    __m256i exp = _mm256_and_si256(_mm256_castps_si256(x), _mm256_set1_epi32(0x7F800000));
    exp = _mm256_srli_epi32(exp, 23);
    __m256 e = _mm256_cvtepi32_ps(exp);
    e = _mm256_sub_ps(e, _mm256_set1_ps(127.0f));
    
    // Extract mantissa and set exponent to 0
    __m256 m = _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x007FFFFF)));
    m = _mm256_or_ps(m, _mm256_castsi256_ps(_mm256_set1_epi32(0x3F800000))); // Set exponent to 0
    
    // Improved minimax polynomial approximation
    __m256 p = _mm256_set1_ps(0.1520749f);
    p = _mm256_fmadd_ps(p, m, _mm256_set1_ps(-0.5659487f));
    p = _mm256_fmadd_ps(p, m, _mm256_set1_ps(0.9614227f));
    p = _mm256_fmadd_ps(p, m, _mm256_set1_ps(-1.0956714f));
    p = _mm256_fmadd_ps(p, m, _mm256_set1_ps(0.9819095f));
    p = _mm256_fmadd_ps(p, m, _mm256_set1_ps(-0.5860944f));
    p = _mm256_fmadd_ps(p, m, _mm256_set1_ps(0.3479320f));
    p = _mm256_fmadd_ps(p, m, _mm256_set1_ps(0.1749663f));
    p = _mm256_fmadd_ps(p, _mm256_sub_ps(m, one), e);
    
    // Multiply by ln(2)
    p = _mm256_mul_ps(p, _mm256_set1_ps(0.6931472f));
    
    // Handle invalid inputs
    return _mm256_and_ps(p, valid_mask);
}

// Improved exp approximation
static inline __m256 exp256_ps(__m256 x) {
    // Clamp input to avoid overflow/underflow
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-88.0f)), _mm256_set1_ps(88.0f));
    
    // Constants for exp approximation
    const __m256 ln2 = _mm256_set1_ps(0.6931472f);
    const __m256 one = _mm256_set1_ps(1.0f);
    
    // n = round(x / ln2)
    __m256 n = _mm256_round_ps(_mm256_div_ps(x, ln2), _MM_FROUND_TO_NEAREST_INT);
    
    // r = x - n * ln2
    x = _mm256_fnmadd_ps(n, ln2, x);
    
    // Polynomial approximation for exp(r)
    __m256 result = one;
    __m256 r = x;
    result = _mm256_add_ps(result, r);
    
    r = _mm256_mul_ps(r, x);
    result = _mm256_fmadd_ps(r, _mm256_set1_ps(0.5f), result);
    
    r = _mm256_mul_ps(r, x);
    result = _mm256_fmadd_ps(r, _mm256_set1_ps(0.166666666f), result);
    
    r = _mm256_mul_ps(r, x);
    result = _mm256_fmadd_ps(r, _mm256_set1_ps(0.041666666f), result);
    
    r = _mm256_mul_ps(r, x);
    result = _mm256_fmadd_ps(r, _mm256_set1_ps(0.008333333f), result);
    
    r = _mm256_mul_ps(r, x);
    result = _mm256_fmadd_ps(r, _mm256_set1_ps(0.001388889f), result);
    
    // Scale by 2^n
    __m256 pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(
                    _mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23));
    
    return _mm256_mul_ps(result, pow2n);
}

// Sanitize gradients by zeroing out non-finite values (NaN, Inf)
void sanitize_gradients(float* data, size_t size) {
    size_t i = 0;
    
    // Process 8 elements at a time using AVX2
    for (; i + VEC_SIZE <= size; i += VEC_SIZE) {
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

// In-place SGD update with SIMD acceleration
void sgd_update_inplace(float* weights, const float* grads, size_t size, float lr) {
    size_t i = 0;
    __m256 lr_vec = _mm256_set1_ps(lr);
    
    // Process 8 elements at a time using AVX2
    for (; i + VEC_SIZE <= size; i += VEC_SIZE) {
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

// Unified tensor transpose function that handles different dimensions
void tensor_transpose(const float* input, float* output, size_t ndim, size_t B, size_t M, size_t N) {
    if (!input || !output) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_transpose\n");
        return;
    }
    
    // Handle different dimensions
    if (ndim == 1) {
        // 1D case: just copy the data (vector to column vector is handled in Python)
        memcpy(output, input, N * sizeof(float));
        return;
    } 
    else if (ndim == 2) {
        // 2D case: matrix transpose
        // For small matrices, use a simple transpose
        if (M * N <= 64) {
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < N; j++) {
                    output[j * M + i] = input[i * N + j];
                }
            }
            return;
        }
        
        // Cache-friendly blocked approach with SIMD
        const size_t block_size = 16; // Larger block size for better cache utilization
        
        #pragma omp parallel for collapse(2) if(M*N > 10000)
        for (size_t i = 0; i < M; i += block_size) {
            for (size_t j = 0; j < N; j += block_size) {
                // Determine block dimensions (handle edge cases)
                size_t block_M = (i + block_size > M) ? (M - i) : block_size;
                size_t block_N = (j + block_size > N) ? (N - j) : block_size;
                
                // Process 8x8 sub-blocks with SIMD when possible
                for (size_t bi = 0; bi < block_M; bi += 8) {
                    for (size_t bj = 0; bj < block_N; bj += 8) {
                        // Check if we have a full 8x8 sub-block
                        if (bi + 8 <= block_M && bj + 8 <= block_N) {
                            // Load 8x8 block with prefetching
                            _mm_prefetch(&input[(i+bi+8)*N + j+bj], _MM_HINT_T0);
                            
                            __m256 row0 = _mm256_loadu_ps(&input[(i+bi+0)*N + j+bj]);
                            __m256 row1 = _mm256_loadu_ps(&input[(i+bi+1)*N + j+bj]);
                            __m256 row2 = _mm256_loadu_ps(&input[(i+bi+2)*N + j+bj]);
                            __m256 row3 = _mm256_loadu_ps(&input[(i+bi+3)*N + j+bj]);
                            __m256 row4 = _mm256_loadu_ps(&input[(i+bi+4)*N + j+bj]);
                            __m256 row5 = _mm256_loadu_ps(&input[(i+bi+5)*N + j+bj]);
                            __m256 row6 = _mm256_loadu_ps(&input[(i+bi+6)*N + j+bj]);
                            __m256 row7 = _mm256_loadu_ps(&input[(i+bi+7)*N + j+bj]);
                            
                            // Transpose 8x8 block using AVX2 intrinsics
                            __m256 t0, t1, t2, t3, t4, t5, t6, t7;
                            
                            // Interleave 32-bit elements
                            t0 = _mm256_unpacklo_ps(row0, row1);
                            t1 = _mm256_unpackhi_ps(row0, row1);
                            t2 = _mm256_unpacklo_ps(row2, row3);
                            t3 = _mm256_unpackhi_ps(row2, row3);
                            t4 = _mm256_unpacklo_ps(row4, row5);
                            t5 = _mm256_unpackhi_ps(row4, row5);
                            t6 = _mm256_unpacklo_ps(row6, row7);
                            t7 = _mm256_unpackhi_ps(row6, row7);
                            
                            // Interleave 64-bit elements
                            row0 = _mm256_shuffle_ps(t0, t2, 0x44);
                            row1 = _mm256_shuffle_ps(t0, t2, 0xEE);
                            row2 = _mm256_shuffle_ps(t1, t3, 0x44);
                            row3 = _mm256_shuffle_ps(t1, t3, 0xEE);
                            row4 = _mm256_shuffle_ps(t4, t6, 0x44);
                            row5 = _mm256_shuffle_ps(t4, t6, 0xEE);
                            row6 = _mm256_shuffle_ps(t5, t7, 0x44);
                            row7 = _mm256_shuffle_ps(t5, t7, 0xEE);
                            
                            // Interleave 128-bit elements
                            t0 = _mm256_permute2f128_ps(row0, row4, 0x20);
                            t1 = _mm256_permute2f128_ps(row1, row5, 0x20);
                            t2 = _mm256_permute2f128_ps(row2, row6, 0x20);
                            t3 = _mm256_permute2f128_ps(row3, row7, 0x20);
                            t4 = _mm256_permute2f128_ps(row0, row4, 0x31);
                            t5 = _mm256_permute2f128_ps(row1, row5, 0x31);
                            t6 = _mm256_permute2f128_ps(row2, row6, 0x31);
                            t7 = _mm256_permute2f128_ps(row3, row7, 0x31);
                            
                            // Store transposed 8x8 block
                            _mm256_storeu_ps(&output[(j+bj+0)*M + i+bi], t0);
                            _mm256_storeu_ps(&output[(j+bj+1)*M + i+bi], t1);
                            _mm256_storeu_ps(&output[(j+bj+2)*M + i+bi], t2);
                            _mm256_storeu_ps(&output[(j+bj+3)*M + i+bi], t3);
                            _mm256_storeu_ps(&output[(j+bj+4)*M + i+bi], t4);
                            _mm256_storeu_ps(&output[(j+bj+5)*M + i+bi], t5);
                            _mm256_storeu_ps(&output[(j+bj+6)*M + i+bi], t6);
                            _mm256_storeu_ps(&output[(j+bj+7)*M + i+bi], t7);
                        } else {
                            // Fallback for partial sub-blocks
                            size_t max_bi = (bi + 8 > block_M) ? block_M - bi : 8;
                            size_t max_bj = (bj + 8 > block_N) ? block_N - bj : 8;
                            
                            for (size_t ii = 0; ii < max_bi; ii++) {
                                for (size_t jj = 0; jj < max_bj; jj++) {
                                    output[(j + bj + jj) * M + (i + bi + ii)] = 
                                        input[(i + bi + ii) * N + (j + bj + jj)];
                                }
                            }
                        }
                    }
                }
            }
        }
    } 
    else if (ndim == 3) {
        // 3D case: batch matrix transpose
        size_t matrix_size = M * N;
        
        #pragma omp parallel for if(B > 4)
        for (size_t b = 0; b < B; b++) {
            // Use direct transpose for each batch instead of recursive call
            const float* input_matrix = input + b * matrix_size;
            float* output_matrix = output + b * matrix_size;
            
            // For small matrices, use a simple transpose
            if (M * N <= 64) {
                for (size_t i = 0; i < M; i++) {
                    for (size_t j = 0; j < N; j++) {
                        output_matrix[j * M + i] = input_matrix[i * N + j];
                    }
                }
                continue;
            }
            
            // Use the same blocked approach as in the 2D case
            const size_t block_size = 16;
            
            for (size_t i = 0; i < M; i += block_size) {
                for (size_t j = 0; j < N; j += block_size) {
                    // Same block processing as in 2D case
                    // ... (identical code to the 2D case block processing)
                    size_t block_M = (i + block_size > M) ? (M - i) : block_size;
                    size_t block_N = (j + block_size > N) ? (N - j) : block_size;
                    
                    for (size_t bi = 0; bi < block_M; bi++) {
                        for (size_t bj = 0; bj < block_N; bj++) {
                            output_matrix[(j + bj) * M + (i + bi)] = 
                                input_matrix[(i + bi) * N + (j + bj)];
                        }
                    }
                }
            }
        }
    } 
    else {
        fprintf(stderr, "Error: Unsupported number of dimensions (%zu) in tensor_transpose\n", ndim);
    }
}

void tensor_transpose_batch(const float* input, float* output, 
                         size_t ndim, size_t B, size_t M, size_t N) {
    // Remove redundant parallel region - already handled in tensor_transpose
    tensor_transpose(input, output, ndim, B, M, N);
}

// Efficient broadcasting of a single row [1, N] → [B, N]
void tensor_broadcast_row(const float* input, float* output, size_t B, size_t N) {
    if (!input || !output) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_broadcast_row\n");
        return;
    }

    #pragma omp parallel for if (B > 4)
    for (size_t b = 0; b < B; ++b) {
        memcpy(output + b * N, input, N * sizeof(float));
    }
}

// Broadcast [B, 1] → [B, N] by repeating each scalar across a row
void tensor_broadcast_col(const float* input, float* output, size_t B, size_t N) {
    if (!input || !output) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_broadcast_col\n");
        return;
    }

    #pragma omp parallel for if(B > 4)
    for (size_t i = 0; i < B; ++i) {
        float val = input[i];
        for (size_t j = 0; j < N; ++j) {
            output[i * N + j] = val;
        }
    }
}

// tensor_unbroadcast_sum_axes: generalized reduction over broadcasted axes
void tensor_unbroadcast_sum_axes(
    const float* grad, float* out,
    const size_t* shape_grad,
    const size_t* shape_out,
    const size_t* strides_grad,
    const size_t* strides_out,
    size_t ndim,
    size_t total_grad,
    size_t total_out
) {
    #pragma omp parallel for
    for (size_t i = 0; i < total_out; ++i) {
        out[i] = 0.0f;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < total_grad; ++i) {
        size_t idx = i;
        size_t out_idx = 0;
        for (int d = 0; d < ndim; ++d) {
            size_t pos = idx / strides_grad[d];
            idx %= strides_grad[d];
            if (shape_out[d] == 1)
                continue;  // broadcasted, skip
            out_idx += pos * strides_out[d];
        }
        #pragma omp atomic
        out[out_idx] += grad[i];
    }
}

// SIMD Elementwise Ops
void tensor_add(const float* a, const float* b, float* out, size_t n) {
    if (!a || !b || !out) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_add\n");
        return;
    }
    
    // Process blocks of 4*VEC_SIZE elements in parallel
    #pragma omp parallel if(n > 10000)
    {
        #pragma omp for
        for (size_t i = 0; i < n; i += 4*VEC_SIZE) {
            if (i + 4*VEC_SIZE <= n) {
                // Prefetch next cache lines
                _mm_prefetch(a + i + 4*VEC_SIZE, _MM_HINT_T0);
                _mm_prefetch(b + i + 4*VEC_SIZE, _MM_HINT_T0);
                
                _mm256_storeu_ps(out + i,             _mm256_add_ps(_mm256_loadu_ps(a + i),             _mm256_loadu_ps(b + i)));
                _mm256_storeu_ps(out + i + VEC_SIZE,   _mm256_add_ps(_mm256_loadu_ps(a + i + VEC_SIZE),   _mm256_loadu_ps(b + i + VEC_SIZE)));
                _mm256_storeu_ps(out + i + 2*VEC_SIZE, _mm256_add_ps(_mm256_loadu_ps(a + i + 2*VEC_SIZE), _mm256_loadu_ps(b + i + 2*VEC_SIZE)));
                _mm256_storeu_ps(out + i + 3*VEC_SIZE, _mm256_add_ps(_mm256_loadu_ps(a + i + 3*VEC_SIZE), _mm256_loadu_ps(b + i + 3*VEC_SIZE)));
            } else {
                // Handle remaining elements
                for (size_t j = i; j < n; j++) {
                    if (j + VEC_SIZE <= n && j % VEC_SIZE == 0) {
                        __m256 va = _mm256_loadu_ps(a + j);
                        __m256 vb = _mm256_loadu_ps(b + j);
                        __m256 vout = _mm256_add_ps(va, vb);
                        _mm256_storeu_ps(out + j, vout);
                        j += VEC_SIZE - 1; // -1 because loop will increment j
                    } else {
                        out[j] = a[j] + b[j];
                    }
                }
            }
        }
    }
}

void tensor_add_grad(const float* dout, const float* a, const float* b, float* da, float* db, size_t n) {
    size_t vec_size = n / VEC_SIZE * VEC_SIZE;
    #pragma omp parallel if(n > 10000)
    {
        #pragma omp for
        for (size_t i = 0; i < vec_size; i += VEC_SIZE) {
            __m256 v_dout = _mm256_loadu_ps(dout + i);
            _mm256_storeu_ps(da + i, _mm256_add_ps(_mm256_loadu_ps(da + i), v_dout));
            _mm256_storeu_ps(db + i, _mm256_add_ps(_mm256_loadu_ps(db + i), v_dout));
        }
        #pragma omp for
        for (size_t i = vec_size; i < n; i++) {
            da[i] += dout[i];
            db[i] += dout[i];
        }
    }
}

void tensor_sub(const float* a, const float* b, float* out, size_t n) {
    if (!a || !b || !out) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_sub\n");
        return;
    }
    
    #pragma omp parallel if(n > 10000)
    {
        #pragma omp for
        for (size_t i = 0; i < n; i += 4*VEC_SIZE) {
            if (i + 4*VEC_SIZE <= n) {
                // Prefetch next cache lines
                _mm_prefetch(a + i + 4*VEC_SIZE, _MM_HINT_T0);
                _mm_prefetch(b + i + 4*VEC_SIZE, _MM_HINT_T0);
                
                _mm256_storeu_ps(out + i,             _mm256_sub_ps(_mm256_loadu_ps(a + i),             _mm256_loadu_ps(b + i)));
                _mm256_storeu_ps(out + i + VEC_SIZE,   _mm256_sub_ps(_mm256_loadu_ps(a + i + VEC_SIZE),   _mm256_loadu_ps(b + i + VEC_SIZE)));
                _mm256_storeu_ps(out + i + 2*VEC_SIZE, _mm256_sub_ps(_mm256_loadu_ps(a + i + 2*VEC_SIZE), _mm256_loadu_ps(b + i + 2*VEC_SIZE)));
                _mm256_storeu_ps(out + i + 3*VEC_SIZE, _mm256_sub_ps(_mm256_loadu_ps(a + i + 3*VEC_SIZE), _mm256_loadu_ps(b + i + 3*VEC_SIZE)));
            } else {
                // Handle remaining elements
                for (size_t j = i; j < n; j++) {
                    if (j + VEC_SIZE <= n && j % VEC_SIZE == 0) {
                        __m256 va = _mm256_loadu_ps(a + j);
                        __m256 vb = _mm256_loadu_ps(b + j);
                        __m256 vout = _mm256_sub_ps(va, vb);
                        _mm256_storeu_ps(out + j, vout);
                        j += VEC_SIZE - 1;
                    } else {
                        out[j] = a[j] - b[j];
                    }
                }
            }
        }
    }
}

void tensor_sub_grad(const float* dout, const float* a, const float* b, float* da, float* db, size_t n) {
    size_t vec_size = n / VEC_SIZE * VEC_SIZE;
    #pragma omp parallel if(n > 10000)
    {
        #pragma omp for
        for (size_t i = 0; i < vec_size; i += VEC_SIZE) {
            __m256 v_dout = _mm256_loadu_ps(dout + i);
            _mm256_storeu_ps(da + i, _mm256_add_ps(_mm256_loadu_ps(da + i), v_dout));
            __m256 v_db = _mm256_sub_ps(_mm256_setzero_ps(), v_dout);
            v_db = _mm256_add_ps(_mm256_loadu_ps(db + i), v_db);
            _mm256_storeu_ps(db + i, v_db);
        }
        #pragma omp for
        for (size_t i = vec_size; i < n; i++) {
            da[i] += dout[i];
            db[i] -= dout[i];
        }
    }
}

void tensor_mul(const float* a, const float* b, float* out, size_t n) {
    if (!a || !b || !out) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_mul\n");
        return;
    }
    
    #pragma omp parallel if(n > 10000)
    {
        #pragma omp for
        for (size_t i = 0; i < n; i += 4*VEC_SIZE) {
            if (i + 4*VEC_SIZE <= n) {
                // Prefetch next cache lines
                _mm_prefetch(a + i + 4*VEC_SIZE, _MM_HINT_T0);
                _mm_prefetch(b + i + 4*VEC_SIZE, _MM_HINT_T0);
                
                _mm256_storeu_ps(out + i,             _mm256_mul_ps(_mm256_loadu_ps(a + i),             _mm256_loadu_ps(b + i)));
                _mm256_storeu_ps(out + i + VEC_SIZE,   _mm256_mul_ps(_mm256_loadu_ps(a + i + VEC_SIZE),   _mm256_loadu_ps(b + i + VEC_SIZE)));
                _mm256_storeu_ps(out + i + 2*VEC_SIZE, _mm256_mul_ps(_mm256_loadu_ps(a + i + 2*VEC_SIZE), _mm256_loadu_ps(b + i + 2*VEC_SIZE)));
                _mm256_storeu_ps(out + i + 3*VEC_SIZE, _mm256_mul_ps(_mm256_loadu_ps(a + i + 3*VEC_SIZE), _mm256_loadu_ps(b + i + 3*VEC_SIZE)));
            } else {
                // Handle remaining elements
                for (size_t j = i; j < n; j++) {
                    if (j + VEC_SIZE <= n && j % VEC_SIZE == 0) {
                        __m256 va = _mm256_loadu_ps(a + j);
                        __m256 vb = _mm256_loadu_ps(b + j);
                        __m256 vout = _mm256_mul_ps(va, vb);
                        _mm256_storeu_ps(out + j, vout);
                        j += VEC_SIZE - 1;
                    } else {
                        out[j] = a[j] * b[j];
                    }
                }
            }
        }
    }
}

void tensor_mul_grad(const float* dout, const float* a, const float* b, float* da, float* db, size_t n) {
    size_t vec_size = n / VEC_SIZE * VEC_SIZE;
    #pragma omp parallel if(n > 10000)
    {
        #pragma omp for
        for (size_t i = 0; i < vec_size; i += VEC_SIZE) {
            __m256 v_dout = _mm256_loadu_ps(dout + i);
            __m256 v_a = _mm256_loadu_ps(a + i);
            __m256 v_b = _mm256_loadu_ps(b + i);
            __m256 v_da = _mm256_mul_ps(v_dout, v_b);
            __m256 v_db = _mm256_mul_ps(v_dout, v_a);
            _mm256_storeu_ps(da + i, _mm256_add_ps(_mm256_loadu_ps(da + i), v_da));
            _mm256_storeu_ps(db + i, _mm256_add_ps(_mm256_loadu_ps(db + i), v_db));
        }
        #pragma omp for
        for (size_t i = vec_size; i < n; i++) {
            da[i] += dout[i] * b[i];
            db[i] += dout[i] * a[i];
        }
    }
}

void tensor_div(const float* a, const float* b, float* out, size_t n) {
    if (!a || !b || !out) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_div\n");
        return;
    }
    
    #pragma omp parallel if(n > 10000)
    {
        #pragma omp for
        for (size_t i = 0; i < n; i += 4*VEC_SIZE) {
            if (i + 4*VEC_SIZE <= n) {
                // Prefetch next cache lines
                _mm_prefetch(a + i + 4*VEC_SIZE, _MM_HINT_T0);
                _mm_prefetch(b + i + 4*VEC_SIZE, _MM_HINT_T0);
                
                _mm256_storeu_ps(out + i,             _mm256_div_ps(_mm256_loadu_ps(a + i),             _mm256_loadu_ps(b + i)));
                _mm256_storeu_ps(out + i + VEC_SIZE,   _mm256_div_ps(_mm256_loadu_ps(a + i + VEC_SIZE),   _mm256_loadu_ps(b + i + VEC_SIZE)));
                _mm256_storeu_ps(out + i + 2*VEC_SIZE, _mm256_div_ps(_mm256_loadu_ps(a + i + 2*VEC_SIZE), _mm256_loadu_ps(b + i + 2*VEC_SIZE)));
                _mm256_storeu_ps(out + i + 3*VEC_SIZE, _mm256_div_ps(_mm256_loadu_ps(a + i + 3*VEC_SIZE), _mm256_loadu_ps(b + i + 3*VEC_SIZE)));
            } else {
                // Handle remaining elements
                for (size_t j = i; j < n; j++) {
                    if (j + VEC_SIZE <= n && j % VEC_SIZE == 0) {
                        __m256 va = _mm256_loadu_ps(a + j);
                        __m256 vb = _mm256_loadu_ps(b + j);
                        __m256 vout = _mm256_div_ps(va, vb);
                        _mm256_storeu_ps(out + j, vout);
                        j += VEC_SIZE - 1;
                    } else {
                        out[j] = a[j] / b[j];
                    }
                }
            }
        }
    }
}

void tensor_div_grad(const float* dout, const float* a, const float* b, float* da, float* db, size_t n) {
    size_t vec_size = n / VEC_SIZE * VEC_SIZE;
    #pragma omp parallel if(n > 10000)
    {
        #pragma omp for
        for (size_t i = 0; i < vec_size; i += VEC_SIZE) {
            __m256 v_dout = _mm256_loadu_ps(dout + i);
            __m256 v_a = _mm256_loadu_ps(a + i);
            __m256 v_b = _mm256_loadu_ps(b + i);
            __m256 v_da = _mm256_div_ps(v_dout, v_b);
            __m256 v_db = _mm256_mul_ps(v_dout, v_a);
            v_db = _mm256_div_ps(v_db, _mm256_mul_ps(v_b, v_b));
            v_db = _mm256_sub_ps(_mm256_setzero_ps(), v_db);
            _mm256_storeu_ps(da + i, _mm256_add_ps(_mm256_loadu_ps(da + i), v_da));
            _mm256_storeu_ps(db + i, _mm256_add_ps(_mm256_loadu_ps(db + i), v_db));
        }
        #pragma omp for
        for (size_t i = vec_size; i < n; i++) {
            da[i] += dout[i] / b[i];
            db[i] -= dout[i] * a[i] / (b[i] * b[i]);
        }
    }
}

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

void tensor_softmax_cross_entropy(const float* logits, 
                                             const float* labels, 
                                             float* loss_out, 
                                             float* probs_out, 
                                             size_t class_count) { 
    if (!logits || !labels || !loss_out || !probs_out) { 
        fprintf(stderr, "Error: NULL pointer passed to tensor_softmax_cross_entropy_with_probs\n"); 
        return; 
    } 
     
    if (class_count > MAX_CLASSES) { 
        fprintf(stderr, "Error: Class count %zu exceeds MAX_CLASSES (%d)\n", class_count, MAX_CLASSES); 
        return; 
    } 
 
    // 1. Find max logit using SIMD 
    __m256 vmax = _mm256_set1_ps(-FLT_MAX); 
    size_t i = 0; 
    for (; i + VEC_SIZE <= class_count; i += VEC_SIZE) { 
        __m256 vlogits = _mm256_loadu_ps(logits + i); 
        vmax = _mm256_max_ps(vmax, vlogits); 
    } 
     
    // Find max value within the vector 
    float max_val = hsum_avx(_mm256_max_ps( 
        _mm256_max_ps( 
            _mm256_permute2f128_ps(vmax, vmax, 1), 
            vmax 
        ), 
        _mm256_permute_ps(vmax, 0x4E) // Shuffle within 128-bit lanes 
    )); 
     
    // Check remaining elements 
    for (; i < class_count; i++) { 
        if (logits[i] > max_val) max_val = logits[i]; 
    } 
 
    // 2. Compute exp, sum, and cross-entropy in a single pass
    __m256 vmax_scalar = _mm256_set1_ps(max_val); 
    __m256 vsum_exp = _mm256_setzero_ps(); 
    __m256 vloss = _mm256_setzero_ps();
    const float epsilon = 1e-7f;
    __m256 vepsilon = _mm256_set1_ps(epsilon);
     
    // First pass: compute exp and sum
    i = 0; 
    for (; i + VEC_SIZE <= class_count; i += VEC_SIZE) { 
        _mm_prefetch(logits + i + VEC_SIZE, _MM_HINT_T0); 
        _mm_prefetch(labels + i + VEC_SIZE, _MM_HINT_T0);
         
        __m256 vlogits = _mm256_loadu_ps(logits + i); 
        __m256 vshifted = _mm256_sub_ps(vlogits, vmax_scalar); 
        __m256 vexp = exp256_ps(vshifted); 
        _mm256_storeu_ps(probs_out + i, vexp); 
        vsum_exp = _mm256_add_ps(vsum_exp, vexp); 
    } 
     
    // Sum the vector elements 
    float sum_exp = hsum_avx(vsum_exp); 
     
    // Process remaining elements 
    for (; i < class_count; i++) { 
        probs_out[i] = expf(logits[i] - max_val); 
        sum_exp += probs_out[i]; 
    } 
 
    // Second pass: normalize and compute cross-entropy in one go
    __m256 vsum_exp_scalar = _mm256_set1_ps(sum_exp); 
    i = 0; 
    for (; i + VEC_SIZE <= class_count; i += VEC_SIZE) { 
        __m256 vprobs = _mm256_loadu_ps(probs_out + i); 
        __m256 vlabels = _mm256_loadu_ps(labels + i);
        
        // Normalize probabilities
        __m256 vnormalized = _mm256_div_ps(vprobs, vsum_exp_scalar); 
        _mm256_storeu_ps(probs_out + i, vnormalized); 
        
        // Compute cross-entropy contribution
        __m256 vsafe_probs = _mm256_add_ps(vnormalized, vepsilon);
        __m256 vlog_probs = log256_ps(vsafe_probs);
        __m256 vweighted = _mm256_mul_ps(vlabels, vlog_probs);
        vloss = _mm256_sub_ps(vloss, vweighted);
    } 
     
    float loss = hsum_avx(vloss);
    
    // Process remaining elements 
    for (; i < class_count; i++) { 
        // Normalize
        probs_out[i] /= sum_exp; 
        
        // Compute cross-entropy contribution
        if (labels[i] > 0.0f) {
            loss -= labels[i] * logf(probs_out[i] + epsilon);
        }
    } 
 
    // Ensure loss is positive
    *loss_out = fabsf(loss); 
}

void tensor_softmax_ce_batch(const float* logits,
                                         const float* labels,
                                         float* losses,
                                         float* probs_out,
                                         size_t batch,
                                         size_t class_count) {
    if (!logits || !labels || !losses || !probs_out) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_op_jit_softmax_ce_with_probs\n");
        return;
    }
    
    #pragma omp parallel for if(batch > 4)
    for (size_t i = 0; i < batch; ++i) {
        const float* logits_row = logits + i * class_count;
        const float* labels_row = labels + i * class_count;
        float* probs_row = probs_out + i * class_count;
        float* loss_ptr = losses + i;

        tensor_softmax_cross_entropy(
            logits_row,
            labels_row,
            loss_ptr,
            probs_row,
            class_count
        );
    }
}

void tensor_softmax_ce_backward( const float* grad_loss, const float* probs, const float* target, float* grad_input, size_t B, size_t C ) {
    size_t total = B * C;
    size_t i = 0;

    // If grad_loss is NULL, scale = 1.0f for all elements
    if (grad_loss == NULL) {
        __m256 one = _mm256_set1_ps(1.0f);
        for (; i + 8 <= total; i += 8) {
            __m256 p = _mm256_loadu_ps(probs + i);
            __m256 t = _mm256_loadu_ps(target + i);
            __m256 g = _mm256_loadu_ps(grad_input + i);

            __m256 diff = _mm256_sub_ps(p, t);
            __m256 res = _mm256_add_ps(g, diff); // g += (p - t) * 1.0f

            _mm256_storeu_ps(grad_input + i, res);
        }
    } else {
        // When grad_loss is not NULL, scale varies per batch element
        for (; i + 8 <= total; i += 8) {
            // Compute which batch indices these 8 elements belong to
            // We assume row-major: element i belongs to batch i / C
            size_t batch_idx0 = (i + 0) / C;
            size_t batch_idx1 = (i + 1) / C;
            size_t batch_idx2 = (i + 2) / C;
            size_t batch_idx3 = (i + 3) / C;
            size_t batch_idx4 = (i + 4) / C;
            size_t batch_idx5 = (i + 5) / C;
            size_t batch_idx6 = (i + 6) / C;
            size_t batch_idx7 = (i + 7) / C;

            // Gather scales per element from grad_loss (scalar values)
            float scales[8] = {
                grad_loss[batch_idx0],
                grad_loss[batch_idx1],
                grad_loss[batch_idx2],
                grad_loss[batch_idx3],
                grad_loss[batch_idx4],
                grad_loss[batch_idx5],
                grad_loss[batch_idx6],
                grad_loss[batch_idx7]
            };

            __m256 scale_vec = _mm256_loadu_ps(scales);
            __m256 p = _mm256_loadu_ps(probs + i);
            __m256 t = _mm256_loadu_ps(target + i);
            __m256 g = _mm256_loadu_ps(grad_input + i);

            __m256 diff = _mm256_sub_ps(p, t);
            __m256 scaled_diff = _mm256_mul_ps(diff, scale_vec);
            __m256 res = _mm256_add_ps(g, scaled_diff);

            _mm256_storeu_ps(grad_input + i, res);
        }
    }

    // Handle remaining elements (tail loop)
    for (; i < total; ++i) {
        float scale = grad_loss == NULL ? 1.0f : grad_loss[i / C];
        grad_input[i] += (probs[i] - target[i]) * scale;
    }
}

void tensor_matmul_batch(const float* __restrict A, const float* __restrict B, float* __restrict C,
                         size_t batch, size_t M, size_t K, size_t N) {
    if (!A || !B || !C) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_matmul_batch\n");
        return;
    }

    size_t total_ops = M * K * N;

    // Handle small matrices directly
    if (total_ops < 10000) {
        #pragma omp parallel for
        for (size_t b = 0; b < batch; ++b) {
            const float* A_b = A + b * M * K;
            float* C_b = C + b * M * N;
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < K; k++) {
                        sum += A_b[i * K + k] * B[k * N + j];
                    }
                    C_b[i * N + j] = sum;
                }
            }
        }
        return;
    }

    // Allocate and transpose B once
    float* __restrict B_T = (float*)_mm_malloc(K * N * sizeof(float), 64);
    if (!B_T) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return;
    }

    // Transpose B into B_T: B_T[n * K + k] = B[k * N + n]
    #pragma omp parallel for
    for (size_t k = 0; k < K; ++k) {
        for (size_t n = 0; n < N; ++n) {
            B_T[n * K + k] = B[k * N + n];
        }
    }

    // Perform batched matrix multiplication using shared B_T
    #pragma omp parallel for
    for (size_t b = 0; b < batch; ++b) {
        const float* __restrict A_b = A + b * M * K;
        float* __restrict C_b = C + b * M * N;

        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                __m256 sum_vec = _mm256_setzero_ps();
                size_t k = 0;

                for (; k + 8 <= K; k += 8) {
                    __m256 a_vec = _mm256_loadu_ps(&A_b[i * K + k]);
                    __m256 b_vec = _mm256_loadu_ps(&B_T[j * K + k]);
                    sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
                }

                float sum = hsum_avx(sum_vec);

                for (; k < K; ++k) {
                    sum += A_b[i * K + k] * B_T[j * K + k];
                }

                C_b[i * N + j] = sum;
            }
        }
    }

    _mm_free(B_T);
}

void tensor_matmul_backward(const float* A, const float* B, const float* grad_out,
                            float* grad_A, float* grad_B,
                            size_t batch, size_t M, size_t K, size_t N, bool accumulate) {
    // B_T and A_T are reused inside
    float* B_T = (float*)_mm_malloc(K * N * sizeof(float), 64);
    float* A_T = (float*)_mm_malloc(M * K * sizeof(float), 64);

    // Transpose B: [K x N] -> [N x K]
    #pragma omp parallel for
    for (size_t k = 0; k < K; ++k)
        for (size_t n = 0; n < N; ++n)
            B_T[n * K + k] = B[k * N + n];

    // Transpose A: [M x K] -> [K x M]
    #pragma omp parallel for
    for (size_t m = 0; m < M; ++m)
        for (size_t k = 0; k < K; ++k)
            A_T[k * M + m] = A[m * K + k];

    // Compute grad_A = grad_out @ B^T
    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < K; ++j) {
                __m256 vsum = _mm256_setzero_ps();
                size_t k = 0;
                for (; k + 7 < N; k += 8) {
                    // Load 8 floats of grad_out
                    __m256 vgrad_out = _mm256_loadu_ps(&grad_out[b * M * N + i * N + k]);
                    // Load 8 floats of B_T (row k)
                    __m256 vB_T = _mm256_loadu_ps(&B_T[(k)*K + j]);
                    vsum = _mm256_fmadd_ps(vgrad_out, vB_T, vsum);
                }
                // Horizontal sum vsum
                float sum = 0.f;
                float buf[8];
                _mm256_storeu_ps(buf, vsum);
                for (int x = 0; x < 8; ++x) sum += buf[x];

                // Handle leftover N elements
                for (; k < N; ++k) {
                    sum += grad_out[b * M * N + i * N + k] * B_T[k * K + j];
                }

                if (accumulate)
                    grad_A[b * M * K + i * K + j] += sum;
                else
                    grad_A[b * M * K + i * K + j] = sum;
            }
        }
    }

    // Compute grad_B = A^T @ grad_out
    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < N; ++j) {
                __m256 vsum = _mm256_setzero_ps();
                size_t k = 0;
                for (; k + 7 < M; k += 8) {
                    // Load 8 floats from A_T (row i)
                    __m256 vA_T = _mm256_loadu_ps(&A_T[i * M + k]);
                    // Load 8 floats from grad_out (row k)
                    float go_buf[8] = {
                        grad_out[b * M * N + (k+0) * N + j],
                        grad_out[b * M * N + (k+1) * N + j],
                        grad_out[b * M * N + (k+2) * N + j],
                        grad_out[b * M * N + (k+3) * N + j],
                        grad_out[b * M * N + (k+4) * N + j],
                        grad_out[b * M * N + (k+5) * N + j],
                        grad_out[b * M * N + (k+6) * N + j],
                        grad_out[b * M * N + (k+7) * N + j],
                    };
                    __m256 vgrad_out = _mm256_loadu_ps(go_buf);
                    vsum = _mm256_fmadd_ps(vA_T, vgrad_out, vsum);
                }
                // Horizontal sum
                float sum = 0.f;
                float buf[8];
                _mm256_storeu_ps(buf, vsum);
                for (int x = 0; x < 8; ++x) sum += buf[x];

                // Handle leftover M elements
                for (; k < M; ++k) {
                    sum += A_T[i * M + k] * grad_out[b * M * N + k * N + j];
                }

                if (accumulate)
                    grad_B[b * K * N + i * N + j] += sum;
                else
                    grad_B[b * K * N + i * N + j] = sum;
            }
        }
    }

    _mm_free(B_T);
    _mm_free(A_T);
}

// Reductions
float tensor_sum(const float* input, size_t len) {
    if (!input) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_sum\n");
        return 0.0f;
    }
    
    float total_sum = 0.0f;
    
    #pragma omp parallel reduction(+:total_sum) if(len > 100000)
    {
        #pragma omp single
        {
            size_t i = 0;
            __m256 vsum1 = _mm256_setzero_ps();
            __m256 vsum2 = _mm256_setzero_ps();
            __m256 vsum3 = _mm256_setzero_ps();
            __m256 vsum4 = _mm256_setzero_ps();
            
            // Process 32 elements at a time
            for (; i + 4*VEC_SIZE <= len; i += 4*VEC_SIZE) {
                _mm_prefetch(input + i + 4*VEC_SIZE, _MM_HINT_T0);
                
                vsum1 = _mm256_add_ps(vsum1, _mm256_loadu_ps(input + i));
                vsum2 = _mm256_add_ps(vsum2, _mm256_loadu_ps(input + i + VEC_SIZE));
                vsum3 = _mm256_add_ps(vsum3, _mm256_loadu_ps(input + i + 2*VEC_SIZE));
                vsum4 = _mm256_add_ps(vsum4, _mm256_loadu_ps(input + i + 3*VEC_SIZE));
            }
            
            // Combine the partial sums
            vsum1 = _mm256_add_ps(vsum1, vsum2);
            vsum3 = _mm256_add_ps(vsum3, vsum4);
            vsum1 = _mm256_add_ps(vsum1, vsum3);
            
            // Process remaining blocks of 8
            for (; i + VEC_SIZE <= len; i += VEC_SIZE) {
                vsum1 = _mm256_add_ps(vsum1, _mm256_loadu_ps(input + i));
            }
            
            // Horizontal sum
            total_sum = hsum_avx(vsum1);
            
            // Process remaining elements
            for (; i < len; i++) {
                total_sum += input[i];
            }
        }
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
