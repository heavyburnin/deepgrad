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
#include <cpuid.h>
#include <x86intrin.h>

#define VEC_SIZE 8
#define MAX_CLASSES 512

// gemmm cache
static float* cached_B_T = NULL;
static size_t cached_B_T_size = 0;

// Check for AVX2 support at runtime
int has_avx2() {
    unsigned int eax, ebx, ecx, edx;

    // Call CPUID with eax = 0 to get highest valid function ID
    if (!__get_cpuid_max(0, NULL)) return 0;

    // Call CPUID with eax=7, ecx=0 to get extended features
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
        return 0;

    return (ebx & bit_AVX2) != 0;
}

// Initialize tensor operations library
int tensor_ops_init() {
    if (!has_avx2()) {
        fprintf(stderr, "Error: AVX2 not supported on this CPU\n");
        return -1;
    }
    
    // Use fewer threads than available cores to avoid oversubscription
    // This prevents thread contention and context switching overhead
    int num_procs = omp_get_num_procs();
    int num_threads = num_procs > 4 ? num_procs - 2 : num_procs;
    
    // For small matrices, too many threads create more overhead than benefit
    // Limit to 4-8 threads for typical deep learning workloads
    if (num_threads > 8) num_threads = 8;
    
    omp_set_num_threads(num_threads);
    
    // Use dynamic scheduling with small chunk size for better load balancing
    #ifdef _OPENMP
    omp_set_schedule(omp_sched_dynamic, 16);
    #endif
    
    fprintf(stderr, "SIMD Tensor Backend initialized with %d threads\n", num_threads);
    
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

// Utility to allocate or reuse aligned memory
static float* get_cached_buffer(float** buf, size_t* current_size, size_t required_size) {
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

void tensor_add(const float* a, const float* b, float* out, size_t n, size_t batch_size) {
    if (!a || !b || !out) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_add_batch\n");
        return;
    }

    // Total number of elements across all batches
    const size_t total_elements = batch_size * n;
    const size_t vec_end = total_elements - (total_elements % 8);

    // Parallelize over batches and elements
    #pragma omp parallel for schedule(static) if (total_elements > 10000)
    for (size_t i = 0; i < vec_end; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
    }

    // Handle remaining elements (if total_elements is not divisible by 8)
    for (size_t i = vec_end; i < total_elements; i++) {
        out[i] = a[i] + b[i];
    }
}

void tensor_add_grad(const float* dout, const float* a, const float* b, float* da, float* db, size_t n, size_t batch_size) {
    if (!dout || !da || !db) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_add_grad_batch\n");
        return;
    }

    // Total number of elements across all batches
    const size_t total_elements = batch_size * n;
    const size_t vec_end = total_elements - (total_elements % 8);

    #pragma omp parallel for schedule(static) if (total_elements > 10000)
    for (size_t i = 0; i < vec_end; i += 8) {
        __m256 v_dout = _mm256_loadu_ps(dout + i);

        __m256 v_da = _mm256_loadu_ps(da + i);
        __m256 v_db = _mm256_loadu_ps(db + i);

        _mm256_storeu_ps(da + i, _mm256_add_ps(v_da, v_dout));
        _mm256_storeu_ps(db + i, _mm256_add_ps(v_db, v_dout));
    }

    // Handle remaining elements
    for (size_t i = vec_end; i < total_elements; i++) {
        da[i] += dout[i];
        db[i] += dout[i];
    }
}

void tensor_sub(const float* a, const float* b, float* out, size_t n, size_t batch_size) {
    if (!a || !b || !out) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_sub_batch\n");
        return;
    }

    size_t total_elements = batch_size * n;
    size_t vec_end = total_elements - (total_elements % 8);

    #pragma omp parallel for schedule(static) if (total_elements > 10000)
    for (size_t i = 0; i < vec_end; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_sub_ps(va, vb));
    }

    for (size_t i = vec_end; i < total_elements; i++) {
        out[i] = a[i] - b[i];
    }
}

void tensor_sub_grad(const float* dout, const float* a, const float* b, float* da, float* db, size_t n, size_t batch_size) {
    if (!dout || !da || !db) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_sub_grad_batch\n");
        return;
    }

    size_t total_elements = batch_size * n;
    size_t vec_end = total_elements - (total_elements % 8);

    #pragma omp parallel for schedule(static) if (total_elements > 10000)
    for (size_t i = 0; i < vec_end; i += 8) {
        __m256 v_dout = _mm256_loadu_ps(dout + i);

        __m256 v_da = _mm256_loadu_ps(da + i);
        __m256 v_db = _mm256_loadu_ps(db + i);

        _mm256_storeu_ps(da + i, _mm256_add_ps(v_da, v_dout));
        _mm256_storeu_ps(db + i, _mm256_sub_ps(v_db, v_dout));
    }

    for (size_t i = vec_end; i < total_elements; i++) {
        da[i] += dout[i];
        db[i] -= dout[i];
    }
}

void tensor_mul(const float* a, const float* b, float* out, size_t n, size_t batch_size) {
    if (!a || !b || !out) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_mul_batch\n");
        return;
    }

    size_t total_elements = batch_size * n;
    size_t vec_end = total_elements - (total_elements % 8);

    #pragma omp parallel for schedule(static) if (total_elements > 10000)
    for (size_t i = 0; i < vec_end; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
    }

    for (size_t i = vec_end; i < total_elements; i++) {
        out[i] = a[i] * b[i];
    }
}

void tensor_mul_grad(const float* dout, const float* a, const float* b, float* da, float* db, size_t n, size_t batch_size) {
    if (!dout || !da || !db) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_mul_grad_batch\n");
        return;
    }

    size_t total_elements = batch_size * n;
    size_t vec_end = total_elements - (total_elements % 8);

    #pragma omp parallel for schedule(static) if (total_elements > 10000)
    for (size_t i = 0; i < vec_end; i += 8) {
        __m256 v_dout = _mm256_loadu_ps(dout + i);
        __m256 v_a = _mm256_loadu_ps(a + i);
        __m256 v_b = _mm256_loadu_ps(b + i);

        __m256 v_da = _mm256_mul_ps(v_dout, v_b);
        __m256 v_db = _mm256_mul_ps(v_dout, v_a);

        _mm256_storeu_ps(da + i, _mm256_add_ps(_mm256_loadu_ps(da + i), v_da));
        _mm256_storeu_ps(db + i, _mm256_add_ps(_mm256_loadu_ps(db + i), v_db));
    }

    for (size_t i = vec_end; i < total_elements; i++) {
        da[i] += dout[i] * b[i];
        db[i] += dout[i] * a[i];
    }
}

void tensor_div(const float* a, const float* b, float* out, size_t n, size_t batch_size) {
    if (!a || !b || !out) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_div_batch\n");
        return;
    }

    size_t total_elements = batch_size * n;
    size_t vec_end = total_elements - (total_elements % 8);

    #pragma omp parallel for schedule(static) if (total_elements > 10000)
    for (size_t i = 0; i < vec_end; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_div_ps(va, vb));
    }

    for (size_t i = vec_end; i < total_elements; i++) {
        out[i] = a[i] / b[i];
    }
}

void tensor_div_grad(const float* dout, const float* a, const float* b, float* da, float* db, size_t n, size_t batch_size) {
    if (!dout || !da || !db) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_div_grad_batch\n");
        return;
    }

    size_t total_elements = batch_size * n;
    size_t vec_end = total_elements - (total_elements % 8);

    #pragma omp parallel for schedule(static) if (total_elements > 10000)
    for (size_t i = 0; i < vec_end; i += 8) {
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

    for (size_t i = vec_end; i < total_elements; i++) {
        da[i] += dout[i] / b[i];
        db[i] -= dout[i] * a[i] / (b[i] * b[i]);
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

void tensor_softmax_ce(const float* logits,
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

void tensor_matmul_free_cache() {
    if (cached_B_T) {
        _mm_free(cached_B_T);
        cached_B_T = NULL;
        cached_B_T_size = 0;
    }
}

void tensor_matmul(const float* __restrict A, const float* __restrict B, float* __restrict C,
                         size_t batch, size_t M, size_t K, size_t N) {
    if (!A || !B || !C) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_matmul\n");
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
    float* __restrict B_T = get_cached_buffer(&cached_B_T, &cached_B_T_size, K * N);
    
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

                //float sum = hsum_avx(sum_vec);

                __attribute__((aligned(32))) float temp[8];
                _mm256_store_ps(temp, sum_vec);
                float sum = temp[0] + temp[1] + temp[2] + temp[3] +
                            temp[4] + temp[5] + temp[6] + temp[7];


                for (; k < K; ++k) {
                    sum += A_b[i * K + k] * B_T[j * K + k];
                }

                C_b[i * N + j] = sum;
            }
        }
    }
    atexit(tensor_matmul_free_cache);
}

void tensor_matmul_backward(const float* A, const float* B, const float* grad_out,
                            float* grad_A, float* grad_B,
                            size_t batch, size_t M, size_t K, size_t N, bool accumulate) {
    float* B_T = get_cached_buffer(&cached_B_T, &cached_B_T_size, K * N);
    if (!B_T) return;

    // Transpose B: [K x N] -> [N x K]
    #pragma omp parallel for
    for (size_t k = 0; k < K; ++k)
        for (size_t n = 0; n < N; ++n)
            B_T[n * K + k] = B[k * N + n];

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
    
    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < N; ++j) {
                __m256 vsum = _mm256_setzero_ps();
                size_t k = 0;
                for (; k + 7 < M; k += 8) {
                    float buf_a[8], buf_g[8];
                    for (int x = 0; x < 8; ++x) {
                        buf_a[x] = A[b * M * K + (k + x) * K + i];         // A[b][k+x][i]
                        buf_g[x] = grad_out[b * M * N + (k + x) * N + j];  // grad_out[b][k+x][j]
                    }
                    __m256 va = _mm256_loadu_ps(buf_a);
                    __m256 vg = _mm256_loadu_ps(buf_g);
                    vsum = _mm256_fmadd_ps(va, vg, vsum);
                }

                float sum = 0.f;
                float buf[8];
                _mm256_storeu_ps(buf, vsum);
                for (int x = 0; x < 8; ++x) sum += buf[x];

                for (; k < M; ++k) {
                    sum += A[b * M * K + k * K + i] * grad_out[b * M * N + k * N + j];
                }

                if (accumulate)
                    grad_B[b * K * N + i * N + j] += sum;
                else
                    grad_B[b * K * N + i * N + j] = sum;
            }
        }
    }
    atexit(tensor_matmul_free_cache);
}
    
// Reductions
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
        float partial_sum = hsum_avx(vsum);

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
