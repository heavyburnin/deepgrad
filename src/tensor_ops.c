// tensor_ops.c

#include "../include/tensor_ops.h"
#include <immintrin.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <string.h>

#define VEC_SIZE 8
#define MAX_CLASSES 512

// AVX2 horizontal sum helper
static inline float hsum_avx(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow  = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// Custom log approximation for AVX
static inline __m256 log256_ps(__m256 x) {
    __m256i exp;
    __m256 frac;
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 half = _mm256_set1_ps(0.5f);
    
    // Extract exponent and normalize fraction to [1,2)
    exp = _mm256_sub_epi32(
        _mm256_srli_epi32(_mm256_castps_si256(x), 23),
        _mm256_set1_epi32(127)
    );
    
    // Build normalized fraction in [1,2)
    frac = _mm256_or_ps(
        _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x007FFFFF))),
        _mm256_castsi256_ps(_mm256_set1_epi32(0x3F800000))
    );
    
    // Polynomial approximation for log(f) where f is in [1,2)
    __m256 y = _mm256_sub_ps(frac, one);
    __m256 y2 = _mm256_mul_ps(y, y);
    __m256 p = _mm256_add_ps(y, _mm256_mul_ps(y2, _mm256_set1_ps(-0.5f)));
    p = _mm256_add_ps(p, _mm256_mul_ps(_mm256_mul_ps(y2, y), _mm256_set1_ps(0.333333f)));
    
    // Combine: log(x) = log(2) * exp + log(frac)
    return _mm256_add_ps(
        _mm256_mul_ps(_mm256_cvtepi32_ps(exp), _mm256_set1_ps(0.693147f)),
        p
    );
}

// SIMD Elementwise Ops
void tensor_add(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    // Process 32 elements at a time (4 AVX2 registers)
    for (; i + 4*VEC_SIZE <= n; i += 4*VEC_SIZE) {
        _mm256_storeu_ps(out + i,             _mm256_add_ps(_mm256_loadu_ps(a + i),             _mm256_loadu_ps(b + i)));
        _mm256_storeu_ps(out + i + VEC_SIZE,   _mm256_add_ps(_mm256_loadu_ps(a + i + VEC_SIZE),   _mm256_loadu_ps(b + i + VEC_SIZE)));
        _mm256_storeu_ps(out + i + 2*VEC_SIZE, _mm256_add_ps(_mm256_loadu_ps(a + i + 2*VEC_SIZE), _mm256_loadu_ps(b + i + 2*VEC_SIZE)));
        _mm256_storeu_ps(out + i + 3*VEC_SIZE, _mm256_add_ps(_mm256_loadu_ps(a + i + 3*VEC_SIZE), _mm256_loadu_ps(b + i + 3*VEC_SIZE)));
    }
    // Process 8 elements at a time
    for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vout = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out + i, vout);
    }
    for (; i < n; i++) out[i] = a[i] + b[i];
}

void tensor_sub(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    // Process 32 elements at a time
    for (; i + 4*VEC_SIZE <= n; i += 4*VEC_SIZE) {
        _mm256_storeu_ps(out + i,             _mm256_sub_ps(_mm256_loadu_ps(a + i),             _mm256_loadu_ps(b + i)));
        _mm256_storeu_ps(out + i + VEC_SIZE,   _mm256_sub_ps(_mm256_loadu_ps(a + i + VEC_SIZE),   _mm256_loadu_ps(b + i + VEC_SIZE)));
        _mm256_storeu_ps(out + i + 2*VEC_SIZE, _mm256_sub_ps(_mm256_loadu_ps(a + i + 2*VEC_SIZE), _mm256_loadu_ps(b + i + 2*VEC_SIZE)));
        _mm256_storeu_ps(out + i + 3*VEC_SIZE, _mm256_sub_ps(_mm256_loadu_ps(a + i + 3*VEC_SIZE), _mm256_loadu_ps(b + i + 3*VEC_SIZE)));
    }
    for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vout = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(out + i, vout);
    }
    for (; i < n; i++) out[i] = a[i] - b[i];
}

void tensor_mul(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    // Process 32 elements at a time
    for (; i + 4*VEC_SIZE <= n; i += 4*VEC_SIZE) {
        _mm256_storeu_ps(out + i,             _mm256_mul_ps(_mm256_loadu_ps(a + i),             _mm256_loadu_ps(b + i)));
        _mm256_storeu_ps(out + i + VEC_SIZE,   _mm256_mul_ps(_mm256_loadu_ps(a + i + VEC_SIZE),   _mm256_loadu_ps(b + i + VEC_SIZE)));
        _mm256_storeu_ps(out + i + 2*VEC_SIZE, _mm256_mul_ps(_mm256_loadu_ps(a + i + 2*VEC_SIZE), _mm256_loadu_ps(b + i + 2*VEC_SIZE)));
        _mm256_storeu_ps(out + i + 3*VEC_SIZE, _mm256_mul_ps(_mm256_loadu_ps(a + i + 3*VEC_SIZE), _mm256_loadu_ps(b + i + 3*VEC_SIZE)));
    }
    for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vout = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(out + i, vout);
    }
    for (; i < n; i++) out[i] = a[i] * b[i];
}

void tensor_div(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    // Process 32 elements at a time
    for (; i + 4*VEC_SIZE <= n; i += 4*VEC_SIZE) {
        _mm256_storeu_ps(out + i,             _mm256_div_ps(_mm256_loadu_ps(a + i),             _mm256_loadu_ps(b + i)));
        _mm256_storeu_ps(out + i + VEC_SIZE,   _mm256_div_ps(_mm256_loadu_ps(a + i + VEC_SIZE),   _mm256_loadu_ps(b + i + VEC_SIZE)));
        _mm256_storeu_ps(out + i + 2*VEC_SIZE, _mm256_div_ps(_mm256_loadu_ps(a + i + 2*VEC_SIZE), _mm256_loadu_ps(b + i + 2*VEC_SIZE)));
        _mm256_storeu_ps(out + i + 3*VEC_SIZE, _mm256_div_ps(_mm256_loadu_ps(a + i + 3*VEC_SIZE), _mm256_loadu_ps(b + i + 3*VEC_SIZE)));
    }
    for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vout = _mm256_div_ps(va, vb);
        _mm256_storeu_ps(out + i, vout);
    }
    for (; i < n; i++) out[i] = a[i] / b[i];
}

// Approximate vectorized exp function
static inline __m256 exp256_ps(__m256 x) {
    // Constants for exp approximation
    const __m256 c1 = _mm256_set1_ps(1.442695f);
    const __m256 c2 = _mm256_set1_ps(0.693147f);
    const __m256 c3 = _mm256_set1_ps(1.0f);
    const __m256 c4 = _mm256_set1_ps(0.5f);
    const __m256 c5 = _mm256_set1_ps(0.166667f);
    const __m256 c6 = _mm256_set1_ps(0.041667f);
    const __m256 c7 = _mm256_set1_ps(88.3762626647949f);
    const __m256 c8 = _mm256_set1_ps(-88.3762626647949f);
    
    // Clamp x to avoid overflow
    x = _mm256_max_ps(_mm256_min_ps(x, c7), c8);
    
    // Calculate exp using polynomial approximation
    __m256 y = _mm256_mul_ps(x, c1);
    __m256 z = _mm256_floor_ps(y);
    x = _mm256_sub_ps(x, _mm256_mul_ps(z, c2));
    
    y = c3;
    y = _mm256_add_ps(y, _mm256_mul_ps(x, c3));
    y = _mm256_add_ps(y, _mm256_mul_ps(_mm256_mul_ps(x, x), c4));
    y = _mm256_add_ps(y, _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(x, x), x), c5));
    y = _mm256_add_ps(y, _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(x, x), x), x), c6));
    
    // Use _mm256_set1_ps(1.0f) for 2^0
    __m256 pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(
                    _mm256_cvtps_epi32(z), _mm256_set1_epi32(127)), 23));
    
    return _mm256_mul_ps(y, pow2n);
}

void tensor_exp(const float* a, float* out, size_t n) {
    size_t i = 0;
    for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vexp = exp256_ps(va);
        _mm256_storeu_ps(out + i, vexp);
    }
    for (; i < n; i++) out[i] = expf(a[i]);
}

void tensor_relu(const float* a, float* out, size_t n) {
    size_t i = 0;
    __m256 zero = _mm256_setzero_ps();
    // Process 32 elements at a time
    for (; i + 4*VEC_SIZE <= n; i += 4*VEC_SIZE) {
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

void tensor_softmax_cross_entropy_with_probs(const float* logits,
                                             const float* labels,
                                             float* loss_out,
                                             float* probs_out,
                                             size_t class_count) {
    assert(class_count <= MAX_CLASSES && "Class count exceeds MAX_CLASSES");

    // 1. Find max logit using SIMD
    __m256 vmax = _mm256_set1_ps(-FLT_MAX);
    size_t i = 0;
    for (; i + VEC_SIZE <= class_count; i += VEC_SIZE) {
        __m256 vlogits = _mm256_loadu_ps(logits + i);
        vmax = _mm256_max_ps(vmax, vlogits);
    }
    
    // Find max value within the vector
    float max_vals[VEC_SIZE];
    _mm256_storeu_ps(max_vals, vmax);
    float max_val = max_vals[0];
    for (int j = 1; j < VEC_SIZE; j++) {
        if (max_vals[j] > max_val) max_val = max_vals[j];
    }
    
    // Check remaining elements
    for (; i < class_count; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    // 2. Compute exp and sum using SIMD
    __m256 vmax_scalar = _mm256_set1_ps(max_val);
    __m256 vsum_exp = _mm256_setzero_ps();
    
    i = 0;
    for (; i + VEC_SIZE <= class_count; i += VEC_SIZE) {
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

    // 3. Normalize to get softmax using SIMD
    __m256 vsum_exp_scalar = _mm256_set1_ps(sum_exp);
    i = 0;
    for (; i + VEC_SIZE <= class_count; i += VEC_SIZE) {
        __m256 vprobs = _mm256_loadu_ps(probs_out + i);
        __m256 vnormalized = _mm256_div_ps(vprobs, vsum_exp_scalar);
        _mm256_storeu_ps(probs_out + i, vnormalized);
    }
    
    // Process remaining elements
    for (; i < class_count; i++) {
        probs_out[i] /= sum_exp;
    }

    // 4. Compute cross-entropy loss
    const float epsilon = 1e-9f;
    __m256 vepsilon = _mm256_set1_ps(epsilon);
    __m256 vloss = _mm256_setzero_ps();
    
    i = 0;
    for (; i + VEC_SIZE <= class_count; i += VEC_SIZE) {
        __m256 vlabels = _mm256_loadu_ps(labels + i);
        __m256 vprobs = _mm256_loadu_ps(probs_out + i);
        __m256 vsafe_probs = _mm256_add_ps(vprobs, vepsilon);
        __m256 vlog_probs = log256_ps(vsafe_probs);  // Use our custom log approximation
        __m256 vweighted = _mm256_mul_ps(vlabels, vlog_probs);
        vloss = _mm256_sub_ps(vloss, vweighted);
    }
    
    float loss = hsum_avx(vloss);
    
    // Process remaining elements
    for (; i < class_count; i++) {
        if (labels[i] > 0.0f) {
            loss -= labels[i] * logf(probs_out[i] + epsilon);
        }
    }

    *loss_out = loss;
}

void tensor_matmul(const float* A, const float* B, float* C, size_t M, size_t K, size_t N) {
    // Transpose B for better memory access patterns
    float* B_transposed = (float*)_mm_malloc(K * N * sizeof(float), 32);
    
    // Transpose B using SIMD where possible
    for (size_t j = 0; j < N; j++) {
        for (size_t k = 0; k < K; k++) {
            B_transposed[j * K + k] = B[k * N + j];
        }
    }
    
    // Zero out C matrix
    memset(C, 0, M * N * sizeof(float));
    
    // Compute matrix multiplication with SIMD
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            __m256 vsum = _mm256_setzero_ps();
            size_t k = 0;
            
            // Process 8 elements at a time
            for (; k + VEC_SIZE <= K; k += VEC_SIZE) {
                __m256 va = _mm256_loadu_ps(&A[i * K + k]);
                __m256 vb = _mm256_loadu_ps(&B_transposed[j * K + k]);
                vsum = _mm256_fmadd_ps(va, vb, vsum);
            }
            
            // Horizontal sum
            float sum = hsum_avx(vsum);
            
            // Process remaining elements
            for (; k < K; k++) {
                sum += A[i * K + k] * B_transposed[j * K + k];
            }
            
            C[i * N + j] = sum;
        }
    }
    
    _mm_free(B_transposed);
}

// Reductions
float tensor_sum(const float* input, size_t len) {
    size_t i = 0;
    __m256 vsum1 = _mm256_setzero_ps();
    __m256 vsum2 = _mm256_setzero_ps();
    __m256 vsum3 = _mm256_setzero_ps();
    __m256 vsum4 = _mm256_setzero_ps();
    
    // Process 32 elements at a time
    for (; i + 4*VEC_SIZE <= len; i += 4*VEC_SIZE) {
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
    float sum = hsum_avx(vsum1);
    
    // Process remaining elements
    for (; i < len; i++) {
        sum += input[i];
    }
    
    return sum;
}

float tensor_mean(const float* input, size_t len) {
    return tensor_sum(input, len) / (float)len;
}

// JIT Batching Wrappers with parallelization
void tensor_op_jit_2in(void (*op)(const float*, const float*, float*, size_t),
                       const float* a, const float* b, float* out,
                       size_t batch, size_t stride) {
    #pragma omp parallel for if(batch > 4)
    for (size_t i = 0; i < batch; ++i) {
        op(a + i * stride, b + i * stride, out + i * stride, stride);
    }
}

void tensor_op_jit_1in(void (*op)(const float*, float*, size_t),
                       const float* a, float* out,
                       size_t batch, size_t stride) {
    #pragma omp parallel for if(batch > 4)
    for (size_t i = 0; i < batch; ++i) {
        op(a + i * stride, out + i * stride, stride);
    }
}

void tensor_op_jit_softmax_ce_with_probs(const float* logits,
                                         const float* labels,
                                         float* losses,
                                         float* probs_out,
                                         size_t batch,
                                         size_t class_count) {
    #pragma omp parallel for if(batch > 4)
    for (size_t i = 0; i < batch; ++i) {
        const float* logits_row = logits + i * class_count;
        const float* labels_row = labels + i * class_count;
        float* probs_row = probs_out + i * class_count;
        float* loss_ptr = losses + i;

        tensor_softmax_cross_entropy_with_probs(
            logits_row,
            labels_row,
            loss_ptr,
            probs_row,
            class_count
        );
    }
}

void tensor_matmul_batch_jit(const float* A, const float* B, float* C,
                         size_t batch, size_t M, size_t K, size_t N) {
    size_t a_stride = M * K;
    size_t c_stride = M * N;

    #pragma omp parallel for if(batch > 2)
    for (size_t i = 0; i < batch; ++i) {
        tensor_matmul(
            A + i * a_stride,
            B,  // shared weight matrix
            C + i * c_stride,
            M, K, N
        );
    }
}