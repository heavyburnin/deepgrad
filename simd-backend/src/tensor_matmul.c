// tensor_matmul.c

#include "tensor_matmul.h"
#include <immintrin.h>
#include <stddef.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <assert.h>

static inline float horizontal_sum(__m256 vec) {
    float tmp[8];
    _mm256_storeu_ps(tmp, vec);
    return tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
}

/* GEMM kernel: computes C = A * (B_T)^T, i.e. A(MxK) * B(KxN) = C(MxN).
 * B_T is expected to be the transpose of the original B (shape N x K).
 * If 'accumulate' is true, results are added into C; otherwise C is overwritten. */
static inline void gemm_avx2_fma(const float* A, const float* B_T, float* C,
                                 size_t M, size_t N, size_t K, bool accumulate) {
    const size_t ldA = K, ldB = K, ldC = N;

    size_t i = 0;
    for (; i + 4 <= M; i += 4) {
        size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            __m256 acc[4][8];
            for (int ii = 0; ii < 4; ++ii)
                for (int jj = 0; jj < 8; ++jj)
                    acc[ii][jj] = _mm256_setzero_ps();

            size_t k = 0;
            for (; k + 8 <= K; k += 8) {
                __m256 a[4] = {
                    _mm256_loadu_ps(&A[(i+0)*ldA + k]),
                    _mm256_loadu_ps(&A[(i+1)*ldA + k]),
                    _mm256_loadu_ps(&A[(i+2)*ldA + k]),
                    _mm256_loadu_ps(&A[(i+3)*ldA + k])
                };
                for (int jj = 0; jj < 8; ++jj) {
                    __m256 b = _mm256_loadu_ps(&B_T[(j+jj)*ldB + k]);
                    for (int ii = 0; ii < 4; ++ii)
                        acc[ii][jj] = _mm256_fmadd_ps(a[ii], b, acc[ii][jj]);
                }
            }

            float out[4][8] = {0};
            for (int ii = 0; ii < 4; ++ii)
                for (int jj = 0; jj < 8; ++jj)
                    out[ii][jj] = horizontal_sum(acc[ii][jj]);

            for (; k < K; ++k) {
                float a[4] = {
                    A[(i+0)*ldA + k],
                    A[(i+1)*ldA + k],
                    A[(i+2)*ldA + k],
                    A[(i+3)*ldA + k]
                };
                for (int jj = 0; jj < 8; ++jj) {
                    float b = B_T[(j+jj)*ldB + k];
                    for (int ii = 0; ii < 4; ++ii)
                        out[ii][jj] += a[ii] * b;
                }
            }
            for (int ii = 0; ii < 4; ++ii)
                for (int jj = 0; jj < 8; ++jj) {
                    if (accumulate) C[(i+ii)*ldC + j+jj] += out[ii][jj];
                    else            C[(i+ii)*ldC + j+jj]  = out[ii][jj];
                }
        }
        // tail N
        for (; j < N; ++j) {
            for (int ii = 0; ii < 4; ++ii) {
                __m256 sum = _mm256_setzero_ps();
                size_t k = 0;
                for (; k + 8 <= K; k += 8) {
                    __m256 a = _mm256_loadu_ps(&A[(i+ii)*ldA + k]);
                    __m256 b = _mm256_loadu_ps(&B_T[j*ldB + k]);
                    sum = _mm256_fmadd_ps(a, b, sum);
                }
                float val = horizontal_sum(sum);
                for (; k < K; ++k)
                    val += A[(i+ii)*ldA + k] * B_T[j*ldB + k];
                if (accumulate) C[(i+ii)*ldC + j] += val;
                else            C[(i+ii)*ldC + j]  = val;
            }
        }
    }
    // tail M
    for (; i < M; ++i) {
        size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            __m256 acc[8];
            for (int jj = 0; jj < 8; ++jj)
                acc[jj] = _mm256_setzero_ps();
            size_t k = 0;
            for (; k + 8 <= K; k += 8) {
                __m256 a = _mm256_loadu_ps(&A[i*ldA + k]);
                for (int jj = 0; jj < 8; ++jj) {
                    __m256 b = _mm256_loadu_ps(&B_T[(j+jj)*ldB + k]);
                    acc[jj] = _mm256_fmadd_ps(a, b, acc[jj]);
                }
            }
            float out[8];
            for (int jj = 0; jj < 8; ++jj)
                out[jj] = horizontal_sum(acc[jj]);
            for (; k < K; ++k) {
                float a = A[i*ldA + k];
                for (int jj = 0; jj < 8; ++jj)
                    out[jj] += a * B_T[(j+jj)*ldB + k];
            }
            for (int jj = 0; jj < 8; ++jj) {
                if (accumulate) C[i*ldC + j+jj] += out[jj];
                else            C[i*ldC + j+jj]  = out[jj];
            }
        }
        for (; j < N; ++j) {
            __m256 sum = _mm256_setzero_ps();
            size_t k = 0;
            for (; k + 8 <= K; k += 8) {
                __m256 a = _mm256_loadu_ps(&A[i*ldA + k]);
                __m256 b = _mm256_loadu_ps(&B_T[j*ldB + k]);
                sum = _mm256_fmadd_ps(a, b, sum);
            }
            float val = horizontal_sum(sum);
            for (; k < K; ++k)
                val += A[i*ldA + k] * B_T[j*ldB + k];
            if (accumulate) C[i*ldC + j] += val;
            else            C[i*ldC + j]  = val;
        }
    }
}

void matmul_forward(const float* A, const float* B, float* C,
                    size_t batch, size_t M, size_t K, size_t N) {
    float* B_T = (float*)_mm_malloc(K * N * sizeof(float), 32);
    if (!B_T) { fprintf(stderr, "Error: malloc B_T failed\n"); return; }
    #pragma omp parallel for collapse(2)
    for (size_t k = 0; k < K; ++k)
        for (size_t n = 0; n < N; ++n)
            B_T[n*K + k] = B[k*N + n];

    #pragma omp parallel for
    for (size_t b = 0; b < batch; ++b) {
        const float* A_b = A + b*M*K;
        float*   C_b = C + b*M*N;
        gemm_avx2_fma(A_b, B_T, C_b, M, N, K, false);
    }
    _mm_free(B_T);
}

void matmul_backward_maybe(const float* A, const float* B, const float* grad_out,
                     float* grad_A, float* grad_B,
                     size_t batch, size_t M, size_t K, size_t N,
                     bool accumulate) {
    // Check pointers (optional, but recommended)
    //fprintf(stderr, "matmul_backward pointers: A=%p, B=%p, grad_out=%p, grad_A=%p\n", A, B, grad_out, grad_A);
    assert(A != NULL && B != NULL && grad_out != NULL && grad_A != NULL);

    // Allocate B_T for transpose of B (shape N x K)
    float* B_T = (float*)_mm_malloc(K * N * sizeof(float), 32);
    if (!B_T) {
        fprintf(stderr, "Error: malloc B_T failed in matmul_backward\n");
        return;
    }

    #pragma omp parallel for collapse(2)
    for (size_t k = 0; k < K; ++k)
        for (size_t n = 0; n < N; ++n)
            B_T[n * K + k] = B[k * N + n];

    #pragma omp parallel for
    for (size_t b = 0; b < batch; ++b) {
        const float* gout_b = grad_out + b * M * N;
        float* gradA_b = grad_A + b * M * K;
        if (!accumulate)
            for (size_t i = 0; i < M * K; ++i)
                gradA_b[i] = 0.0f;
        gemm_avx2_fma(gout_b, B_T, gradA_b, M, K, N, accumulate);
    }
    _mm_free(B_T);

    if (grad_B) {
        #pragma omp parallel for collapse(2)
        for (size_t k = 0; k < K; ++k) {
            for (size_t n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (size_t b = 0; b < batch; ++b)
                    for (size_t m = 0; m < M; ++m)
                        sum += A[b * M * K + m * K + k] * grad_out[b * M * N + m * N + n];
                size_t idx = k * N + n;
                if (accumulate) grad_B[idx] += sum;
                else            grad_B[idx]  = sum;
            }
        }
    }
}

void matmul_backd(const float* A, const float* B, const float* grad_out,
                     float* grad_A, float* grad_B,
                     size_t batch, size_t M, size_t K, size_t N,
                     bool accumulate) {
    // Check only required pointers
    assert(A != NULL && B != NULL && grad_out != NULL);
    
    // Allocate B_T for transpose of B (shape N x K)
    float* B_T = (float*)_mm_malloc(K * N * sizeof(float), 32);
    if (!B_T) {
        fprintf(stderr, "Error: malloc B_T failed in matmul_backward\n");
        return;
    }

    #pragma omp parallel for collapse(2)
    for (size_t k = 0; k < K; ++k)
        for (size_t n = 0; n < N; ++n)
            B_T[n * K + k] = B[k * N + n];

    // Only compute gradient for A if requested
    if (grad_A != NULL) {
        #pragma omp parallel for
        for (size_t b = 0; b < batch; ++b) {
            const float* gout_b = grad_out + b * M * N;
            float* gradA_b = grad_A + b * M * K;
            if (!accumulate) {
                // Only zero if we're not accumulating
                #pragma omp simd
                for (size_t i = 0; i < M * K; ++i)
                    gradA_b[i] = 0.0f;
            }
            gemm_avx2_fma(gout_b, B_T, gradA_b, M, K, N, accumulate);
        }
    }

    // Compute gradient for B if requested
    if (grad_B != NULL) {
        #pragma omp parallel for collapse(2)
        for (size_t k = 0; k < K; ++k) {
            for (size_t n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (size_t b = 0; b < batch; ++b)
                    for (size_t m = 0; m < M; ++m)
                        sum += A[b * M * K + m * K + k] * grad_out[b * M * N + m * N + n];
                size_t idx = k * N + n;
                if (accumulate) grad_B[idx] += sum;
                else            grad_B[idx]  = sum;
            }
        }
    }

    _mm_free(B_T);
}

void matmul_backward(const float* A, const float* B, const float* grad_out,
                     float* grad_A, float* grad_B,
                     size_t batch, size_t M, size_t K, size_t N,
                     bool accumulate) {
    // printf("matmul_backward: A=%p, B=%p, grad_out=%p, grad_A=%p, grad_B=%p, batch=%zu, M=%zu, K=%zu, N=%zu, accumulate=%d\n",
           // A, B, grad_out, grad_A, grad_B, batch, M, K, N, accumulate);
    
    // Check required pointers
    assert(A != NULL && B != NULL);
    assert(grad_A != NULL || grad_B != NULL); // At least one gradient must be computed
    
    // Allocate B_T for transpose of B (shape N x K)
    float* B_T = (float*)_mm_malloc(K * N * sizeof(float), 32);
    if (!B_T) {
        fprintf(stderr, "Error: malloc B_T failed in matmul_backward\n");
        return;
    }

    #pragma omp parallel for collapse(2)
    for (size_t k = 0; k < K; ++k)
        for (size_t n = 0; n < N; ++n)
            B_T[n * K + k] = B[k * N + n];

    // Compute gradient for A if requested
    if (grad_A != NULL) {
        if (grad_out == NULL) {
            fprintf(stderr, "Error: grad_out is NULL but grad_A is requested in matmul_backward\n");
            _mm_free(B_T);
            return;
        }
        #pragma omp parallel for
        for (size_t b = 0; b < batch; ++b) {
            const float* gout_b = grad_out + b * M * N;
            float* gradA_b = grad_A + b * M * K;
            if (!accumulate) {
                #pragma omp simd
                for (size_t i = 0; i < M * K; ++i)
                    gradA_b[i] = 0.0f;
            }
            // printf("matmul_backward: Computing grad_A, batch=%zu, shape=(%zu, %zu)\n", b, M, K);
            gemm_avx2_fma(gout_b, B_T, gradA_b, M, K, N, accumulate);
        }
    }

    // Compute gradient for B if requested
    if (grad_B != NULL) {
        if (grad_out == NULL) {
            fprintf(stderr, "Error: grad_out is NULL but grad_B is requested in matmul_backward\n");
            _mm_free(B_T);
            return;
        }
        #pragma omp parallel for collapse(2)
        for (size_t k = 0; k < K; ++k) {
            for (size_t n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (size_t b = 0; b < batch; ++b)
                    for (size_t m = 0; m < M; ++m)
                        sum += A[b * M * K + m * K + k] * grad_out[b * M * N + m * N + n];
                size_t idx = k * N + n;
                if (accumulate) grad_B[idx] += sum;
                else            grad_B[idx]  = sum;
            }
        }
    }

    _mm_free(B_T);
}