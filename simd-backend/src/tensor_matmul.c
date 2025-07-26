// tensor_matmul.c

#include "tensor_matmul.h"
#include <immintrin.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define TILE_M 4
#define TILE_N 8

// Optimized horizontal sum for AVX2
static inline float horizontal_sum(__m256 vec) {
    __m128 vlow = _mm256_castps256_ps128(vec);
    __m128 vhigh = _mm256_extractf128_ps(vec, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    vlow = _mm_add_ps(vlow, _mm_movehl_ps(vlow, vlow));
    vlow = _mm_add_ss(vlow, _mm_movehdup_ps(vlow));
    return _mm_cvtss_f32(vlow);
}

// FMA + AVX2 + tiling GEMM: C[M×N] = A[M×K] * B[K×N]
void gemm_avx2_fma(const float* restrict A, const float* restrict B, float* restrict C,
                   size_t M, size_t N, size_t K, bool accumulate) {
    const size_t ldA = K, ldB = N, ldC = N;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < M; i += TILE_M) {
        for (size_t j = 0; j < N; j += TILE_N) {
            __m256 acc[TILE_M] = {0};
            for (int ii = 0; ii < TILE_M; ++ii)
                acc[ii] = _mm256_setzero_ps();

            for (size_t k = 0; k < K; ++k) {
                __m256 b_vec;
                const float* b_ptr = &B[k * ldB + j];

                if (j + TILE_N <= N) {
                    b_vec = _mm256_loadu_ps(b_ptr);
                } else {
                    float tmp[TILE_N] = {0};
                    for (size_t jj = 0; jj < TILE_N && (j + jj) < N; ++jj)
                        tmp[jj] = b_ptr[jj];
                    b_vec = _mm256_loadu_ps(tmp);
                }

                for (int ii = 0; ii < TILE_M; ++ii) {
                    if (i + ii >= M) break;
                    float a_val = A[(i + ii) * ldA + k];
                    __m256 a_bcast = _mm256_set1_ps(a_val);
                    acc[ii] = _mm256_fmadd_ps(a_bcast, b_vec, acc[ii]);
                }
            }

            for (int ii = 0; ii < TILE_M; ++ii) {
                if (i + ii >= M) break;
                float* c_ptr = &C[(i + ii) * ldC + j];

                if (j + TILE_N <= N) {
                    if (accumulate) {
                        __m256 c_old = _mm256_loadu_ps(c_ptr);
                        acc[ii] = _mm256_add_ps(acc[ii], c_old);
                    }
                    _mm256_storeu_ps(c_ptr, acc[ii]);
                } else {
                    float tmp[TILE_N];
                    _mm256_storeu_ps(tmp, acc[ii]);
                    for (size_t jj = 0; jj < TILE_N && (j + jj) < N; ++jj) {
                        if (accumulate)
                            c_ptr[jj] += tmp[jj];
                        else
                            c_ptr[jj] = tmp[jj];
                    }
                }
            }
        }
    }
}

void matmul_forward(const float* A, const float* B, float* C,
                    size_t batch, size_t M, size_t K, size_t N) {
    if (batch == 0 || M == 0 || K == 0 || N == 0) {
        memset(C, 0, batch * M * N * sizeof(float));
        return;
    }

    #pragma omp parallel for schedule(static)
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
    if (batch == 0 || M == 0 || K == 0 || N == 0) {
        if (grad_A && !accumulate) memset(grad_A, 0, batch * M * K * sizeof(float));
        if (grad_B && !accumulate) memset(grad_B, 0, batch * K * N * sizeof(float));
        return;
    }

    #pragma omp parallel
    {
        float* A_T = grad_B ? (float*)_mm_malloc(K * M * sizeof(float), 32) : NULL;
        float* B_T = grad_A ? (float*)_mm_malloc(N * K * sizeof(float), 32) : NULL;

        #pragma omp for schedule(static)
        for (size_t b = 0; b < batch; ++b) {
            const float* A_b = A + b * M * K;
            const float* B_b = B + b * K * N;
            const float* G_b = grad_out + b * M * N;

            if (grad_A) {
                float* gradA_b = grad_A + b * M * K;

                for (size_t n = 0; n < N; ++n)
                    for (size_t k = 0; k < K; ++k)
                        B_T[n * K + k] = B_b[k * N + n];

                gemm_avx2_fma(G_b, B_T, gradA_b, M, K, N, accumulate);
            }

            if (grad_B) {
                float* gradB_b = grad_B + b * K * N;

                for (size_t k = 0; k < K; ++k)
                    for (size_t m = 0; m < M; ++m)
                        A_T[k * M + m] = A_b[m * K + k];

                gemm_avx2_fma(A_T, G_b, gradB_b, K, N, M, accumulate);
            }
        }

        if (A_T) _mm_free(A_T);
        if (B_T) _mm_free(B_T);
    }
}
