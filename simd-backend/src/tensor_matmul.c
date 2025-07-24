#include "tensor_matmul.h"
#include <immintrin.h>
#include <stddef.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

// horizontal sum for AVX2
static inline float horizontal_sum(__m256 vec) {
    __m128 vlow = _mm256_castps256_ps128(vec);
    __m128 vhigh = _mm256_extractf128_ps(vec, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(2, 3, 0, 1));
    vlow = _mm_add_ps(vlow, shuf);
    shuf = _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(1, 0, 3, 2));
    vlow = _mm_add_ps(vlow, shuf);
    return _mm_cvtss_f32(vlow);
}

/* GEMM kernel: C = A * B, where:
 * A: (M x K)
 * B: (K x N)
 * C: (M x N)
 * All are row-major, and B is NOT transposed.
 */
void gemm_avx2_fma(const float* A, const float* B, float* C,
                   size_t M, size_t N, size_t K, bool accumulate) {
    if (M == 0 || N == 0 || K == 0) return;

    const size_t ldA = K, ldB = N, ldC = N;

    // Tile sizes: 4 rows of A, 8 cols of B
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; i += 4) {
        for (size_t j = 0; j < N; j += 8) {

            float out[4][8] = {0};

            for (size_t k = 0; k < K; ++k) {
                // Load A rows as scalars
                float a_vals[4] = {0};
                for (size_t ii = 0; ii < 4 && i + ii < M; ++ii) {
                    a_vals[ii] = A[(i + ii) * ldA + k];
                }

                // Load 8 floats from the k-th row of B
                __m256 b_vec = _mm256_setzero_ps();
                if (j + 8 <= N) {
                    b_vec = _mm256_loadu_ps(&B[k * ldB + j]);
                } else {
                    // handle tail
                    float b_tail[8] = {0};
                    for (size_t jj = 0; jj < 8 && j + jj < N; ++jj) {
                        b_tail[jj] = B[k * ldB + j + jj];
                    }
                    b_vec = _mm256_loadu_ps(b_tail);
                }

                // Multiply and accumulate
                for (size_t ii = 0; ii < 4 && i + ii < M; ++ii) {
                    __m256 a_broadcast = _mm256_set1_ps(a_vals[ii]);
                    __m256 c = _mm256_mul_ps(a_broadcast, b_vec);

                    // Add to output
                    float c_temp[8];
                    _mm256_storeu_ps(c_temp, c);
                    for (size_t jj = 0; jj < 8 && j + jj < N; ++jj) {
                        out[ii][jj] += c_temp[jj];
                    }
                }
            }

            // Store results
            for (size_t ii = 0; ii < 4 && i + ii < M; ++ii) {
                for (size_t jj = 0; jj < 8 && j + jj < N; ++jj) {
                    size_t idx = (i + ii) * ldC + (j + jj);
                    if (accumulate) {
                        C[idx] += out[ii][jj];
                    } else {
                        C[idx] = out[ii][jj];
                    }
                }
            }
        }
    }
}

void matmul_forward(const float* A, const float* B, float* C,
                    size_t batch, size_t M, size_t K, size_t N) {
    if (batch == 0 || M == 0 || N == 0 || K == 0) {
        memset(C, 0, batch * M * N * sizeof(float));
        return;
    }

    #pragma omp parallel for
    for (size_t b = 0; b < batch; ++b) {
        const float* A_b = A + b * M * K;
        const float* B_b = B + b * K * N;
        float* C_b = C + b * M * N;
        gemm_avx2_fma(A_b, B_b, C_b, M, N, K, false);
    }
}

void matmul_backward(const float* A, const float* B, const float* grad_out,
                     float* grad_A, float* grad_B,
                     size_t batch, size_t M, size_t K, size_t N,
                     bool accumulate) {
    if (batch == 0 || M == 0 || N == 0 || K == 0) {
        if (grad_A && !accumulate) memset(grad_A, 0, batch * M * K * sizeof(float));
        if (grad_B && !accumulate) memset(grad_B, 0, batch * K * N * sizeof(float));
        return;
    }

    // Compute grad_A: grad_A = grad_out @ B^T
    if (grad_A) {
        #pragma omp parallel for
        for (size_t b = 0; b < batch; ++b) {
            const float* gout_b = grad_out + b * M * N;
            const float* B_b = B + b * K * N;
            float* gradA_b = grad_A + b * M * K;

            // Transpose B_b: (K, N) -> (N, K)
            float* B_b_T = (float*)_mm_malloc(N * K * sizeof(float), 32);
            for (size_t n = 0; n < N; ++n) {
                for (size_t k = 0; k < K; ++k) {
                    B_b_T[n * K + k] = B_b[k * N + n];
                }
            }

            gemm_avx2_fma(gout_b, B_b_T, gradA_b, M, K, N, accumulate);
            _mm_free(B_b_T);
        }
    }

    // Compute grad_B: grad_B = A^T @ grad_out
    if (grad_B) {
        #pragma omp parallel for
        for (size_t b = 0; b < batch; ++b) {
            const float* A_b = A + b * M * K;
            const float* gout_b = grad_out + b * M * N;
            float* gradB_b = grad_B + b * K * N;

            // Transpose A_b: (M, K) -> (K, M)
            float* A_b_T = (float*)_mm_malloc(K * M * sizeof(float), 32);
            for (size_t k = 0; k < K; ++k) {
                for (size_t m = 0; m < M; ++m) {
                    A_b_T[k * M + m] = A_b[m * K + k]; // Fixed index
                }
            }

            gemm_avx2_fma(A_b_T, gout_b, gradB_b, K, N, M, accumulate);
            _mm_free(A_b_T);
        }
    }
}