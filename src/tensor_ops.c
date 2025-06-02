// tensor_ops.c

#include "../include/tensor_ops.h"
#include <immintrin.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#define VEC_SIZE 8
#define MAX_CLASSES 512

// SIMD Elementwise Ops
void tensor_add(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
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
    for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vout = _mm256_div_ps(va, vb);
        _mm256_storeu_ps(out + i, vout);
    }
    for (; i < n; i++) out[i] = a[i] / b[i];
}

void tensor_exp(const float* a, float* out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = expf(a[i]);
}

void tensor_relu(const float* a, float* out, size_t n) {
    size_t i = 0;
    __m256 zero = _mm256_setzero_ps();
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

    float max_val = -FLT_MAX;

    // 1. Find max logit
    for (size_t i = 0; i < class_count; ++i)
        if (logits[i] > max_val) max_val = logits[i];

    // 2. Compute exp
    float sum_exp = 0.0f;
    for (size_t i = 0; i < class_count; ++i) {
        probs_out[i] = expf(logits[i] - max_val);
        sum_exp += probs_out[i];
    }

    // 3. Normalize to get softmax
    for (size_t i = 0; i < class_count; ++i) {
        probs_out[i] /= sum_exp;
    }

    // 4. Compute cross-entropy loss
    float loss = 0.0f;
    for (size_t i = 0; i < class_count; ++i) {
        if (labels[i] > 0.0f) {
            loss -= labels[i] * logf(probs_out[i] + 1e-9f);
        }
    }

    *loss_out = loss;
}

void tensor_matmul(const float* A, const float* B, float* C, size_t M, size_t K, size_t N) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            size_t k = 0;

            for (; k + VEC_SIZE <= K; k += VEC_SIZE) {
                __m256 va = _mm256_loadu_ps(&A[i * K + k]);

                float b_vals[VEC_SIZE];
                for (int x = 0; x < VEC_SIZE; ++x)
                    b_vals[x] = B[(k + x) * N + j];  // Column-major access workaround

                __m256 vb = _mm256_loadu_ps(b_vals);
                __m256 vmul = _mm256_mul_ps(va, vb);

                float temp[VEC_SIZE];
                _mm256_storeu_ps(temp, vmul);
                for (int x = 0; x < VEC_SIZE; ++x)
                    sum += temp[x];
            }

            for (; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];

            C[i * N + j] = sum;
        }
    }
}

void transpose_B(const float* B, float* B_T, size_t K, size_t N) {
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < N; ++j) {
            B_T[j * K + k] = B[k * N + j];  // Transpose from [K x N] to [N x K]
        }
    }
}

// B_T must be pre-transposed: B_T[j * K + k] = B[k * N + j]
void tensor_matmul_batch_transposedB(const float* A, const float* B_T, float* C,
                                     size_t batch, size_t M, size_t K, size_t N) {
    for (size_t b = 0; b < batch; ++b) {
        const float* Ab = A + b * M * K;
        float* Cb = C + b * M * N;

        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                __m256 vsum = _mm256_setzero_ps();
                size_t k = 0;

                for (; k + VEC_SIZE <= K; k += VEC_SIZE) {
                    // Load 8 A values
                    __m256 va = _mm256_loadu_ps(&Ab[i * K + k]);

                    // Load 8 B_T values (B_T is [N x K] row-major)
                    __m256 vb = _mm256_loadu_ps(&B_T[j * K + k]);

                    vsum = _mm256_fmadd_ps(va, vb, vsum);
                }

                float sum = 0.0f;
                float temp[VEC_SIZE];
                _mm256_storeu_ps(temp, vsum);
                for (int x = 0; x < VEC_SIZE; ++x)
                    sum += temp[x];

                for (; k < K; ++k)
                    sum += Ab[i * K + k] * B_T[j * K + k];

                Cb[i * N + j] = sum;
            }
        }
    }
}

// Reductions
float tensor_sum(const float* input, size_t len) {
    float sum = 0.0f;
    size_t i = 0;
    for (; i + VEC_SIZE <= len; i += VEC_SIZE) {
        __m256 v = _mm256_loadu_ps(input + i);
        float tmp[VEC_SIZE];
        _mm256_storeu_ps(tmp, v);
        for (int j = 0; j < VEC_SIZE; j++) sum += tmp[j];
    }
    for (; i < len; ++i) sum += input[i];
    return sum;
}

float tensor_mean(const float* input, size_t len) {
    return tensor_sum(input, len) / (float)len;
}


// JIT Batching Wrappers
void tensor_op_jit_2in(void (*op)(const float*, const float*, float*, size_t),
                       const float* a, const float* b, float* out,
                       size_t batch, size_t stride) {
    for (size_t i = 0; i < batch; ++i) {
        op(a + i * stride, b + i * stride, out + i * stride, stride);
    }
}

void tensor_op_jit_1in(void (*op)(const float*, float*, size_t),
                       const float* a, float* out,
                       size_t batch, size_t stride) {
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

void tensor_matmul_batch(const float* A, const float* B, float* C,
                         size_t batch, size_t M, size_t K, size_t N) {
    size_t a_stride = M * K;
    size_t c_stride = M * N;

    for (size_t i = 0; i < batch; ++i) {
        tensor_matmul(
            A + i * a_stride,
            B,  // shared weight matrix
            C + i * c_stride,
            M, K, N
        );
    }
}

// B: original shape [K x N], row-major (column-major access pattern)
// A: shape [batch x M x K]
// C: shape [batch x M x N]
void tensor_matmul_batch_jit(const float* A, const float* B, float* C,
                             size_t batch, size_t M, size_t K, size_t N) {
    // Step 1: Transpose B (from [K x N] to [N x K])
    float* B_T = (float*)aligned_alloc(32, sizeof(float) * K * N);
    if (!B_T) {
        // Handle allocation failure (optional: log or assert)
        return;
    }

    for (size_t k = 0; k < K; ++k)
        for (size_t j = 0; j < N; ++j)
            B_T[j * K + k] = B[k * N + j];

    // Step 2: Call optimized vectorized batched matmul
    tensor_matmul_batch_transposedB(A, B_T, C, batch, M, K, N);

    // Step 3: Clean up
    free(B_T);
}