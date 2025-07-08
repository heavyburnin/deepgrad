// tensor_matmul.c

#include "tensor_matmul.h"
#include "tensor_utils.h"   // for get_cached_buffer()
#include <immintrin.h>      // AVX intrinsics
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <mm_malloc.h>
#include <omp.h>
#include <assert.h>

#define TILE_M 64
#define TILE_N 64
#define TILE_K 64

// Cached buffers for transpose and temporary storage
static float* cached_B_T = NULL;
static size_t cached_B_T_size = 0;

static float* cached_grad_out_T = NULL;
static size_t cached_grad_out_T_size = 0;

// Cache to avoid redundant transposes
static const float* last_B_ptr = NULL;
static size_t last_K = 0;
static size_t last_N = 0;

static bool matmul_cache_cleanup_registered = false;

void tensor_matmul_free_cache() {
    if (cached_B_T) {
        _mm_free(cached_B_T);
        cached_B_T = NULL;
        cached_B_T_size = 0;
    }
    if (cached_grad_out_T) {
        _mm_free(cached_grad_out_T);
        cached_grad_out_T = NULL;
        cached_grad_out_T_size = 0;
    }
    last_B_ptr = NULL;
    last_K = 0;
    last_N = 0;
}

void matmul_forward(
    const float* A, const float* B,
    float* C,
    size_t batch, size_t M, size_t K, size_t N
) {
    size_t total_ops = M * K * N;

    // For small matmuls, fallback to scalar loop to avoid overhead
    if (batch * total_ops < 10000) {
        #pragma omp parallel for
        for (size_t b = 0; b < batch; ++b) {
            const float* A_b = A + b * M * K;
            float* C_b = C + b * M * N;
            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < K; ++k)
                        sum += A_b[i * K + k] * B[k * N + j];
                    C_b[i * N + j] = sum;
                }
            }
        }
        return;
    }

    // Allocate and transpose B (KxN -> NxK) for better memory access in AVX2
    float* B_T = get_cached_buffer(&cached_B_T, &cached_B_T_size, K * N);
    if (!B_T) {
        fprintf(stderr, "Error: Memory allocation failed for B_T\n");
        return;
    }

    if (B != last_B_ptr || K != last_K || N != last_N) {
        last_B_ptr = B;
        last_K = K;
        last_N = N;

        #pragma omp parallel for collapse(2)
        for (size_t k = 0; k < K; ++k)
            for (size_t n = 0; n < N; ++n)
                B_T[n * K + k] = B[k * N + n];
    }

    // Main tiled AVX2 matmul with OpenMP parallelization over batch and tiles
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t i0 = 0; i0 < M; i0 += TILE_M) {
            for (size_t j0 = 0; j0 < N; j0 += TILE_N) {
                for (size_t k0 = 0; k0 < K; k0 += TILE_K) {
                    const float* A_b = A + b * M * K;
                    float* C_b = C + b * M * N;

                    size_t i_max = (i0 + TILE_M > M) ? M : i0 + TILE_M;
                    size_t j_max = (j0 + TILE_N > N) ? N : j0 + TILE_N;
                    size_t k_max = (k0 + TILE_K > K) ? K : k0 + TILE_K;

                    for (size_t i = i0; i < i_max; ++i) {
                        for (size_t j = j0; j < j_max; ++j) {
                            __m256 sum_vec = _mm256_setzero_ps();
                            size_t k = k0;
                            for (; k + 8 <= k_max; k += 8) {
                                __m256 a_vec = _mm256_loadu_ps(&A_b[i * K + k]);
                                __m256 b_vec = _mm256_loadu_ps(&B_T[j * K + k]);
                                sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
                            }
                            float temp[8];
                            _mm256_storeu_ps(temp, sum_vec);
                            float sum = temp[0] + temp[1] + temp[2] + temp[3] +
                                        temp[4] + temp[5] + temp[6] + temp[7];
                            for (; k < k_max; ++k)
                                sum += A_b[i * K + k] * B_T[j * K + k];

                            if (k0 == 0)
                                C_b[i * N + j] = sum;
                            else
                                C_b[i * N + j] += sum;
                        }
                    }
                }
            }
        }
    }
}

void matmul_backward(
    const float* A, const float* B, const float* grad_out,
    float* grad_A, float* grad_B,
    size_t batch, size_t M, size_t K, size_t N,
    bool accumulate
) {
    // Allocate and transpose B_T (KxN -> NxK)
    float* B_T = get_cached_buffer(&cached_B_T, &cached_B_T_size, K * N);
    if (!B_T) {
        fprintf(stderr, "Error: Memory allocation failed for B_T\n");
        return;
    }

    if (B != last_B_ptr || K != last_K || N != last_N) {
        last_B_ptr = B;
        last_K = K;
        last_N = N;

        #pragma omp parallel for collapse(2)
        for (size_t k = 0; k < K; ++k)
            for (size_t n = 0; n < N; ++n)
                B_T[n * K + k] = B[k * N + n];
    }

    // Allocate and transpose grad_out_T (batch x M x N -> batch x N x M) for grad_B calculation
    float* grad_out_T = NULL;
    if (grad_B != NULL) {
        grad_out_T = get_cached_buffer(&cached_grad_out_T, &cached_grad_out_T_size, batch * N * M);
        if (!grad_out_T) {
            fprintf(stderr, "Error: Memory allocation failed for grad_out_T\n");
            return;
        }

        #pragma omp parallel for collapse(3)
        for (size_t b = 0; b < batch; ++b) {
            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    grad_out_T[b * N * M + j * M + i] = grad_out[b * M * N + i * N + j];
                }
            }
        }
    }

    // Compute grad_A: tiled AVX2 kernel (batch x M x K)
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t i0 = 0; i0 < M; i0 += TILE_M) {
            for (size_t j0 = 0; j0 < K; j0 += TILE_K) {
                for (size_t k0 = 0; k0 < N; k0 += TILE_N) {
                    size_t i_max = (i0 + TILE_M > M) ? M : i0 + TILE_M;
                    size_t j_max = (j0 + TILE_K > K) ? K : j0 + TILE_K;
                    size_t k_max = (k0 + TILE_N > N) ? N : k0 + TILE_N;

                    for (size_t i = i0; i < i_max; ++i) {
                        for (size_t j = j0; j < j_max; ++j) {
                            __m256 vsum_a = _mm256_setzero_ps();
                            size_t k = k0;
                            for (; k + 8 <= k_max; k += 8) {
                                __m256 vgrad_out = _mm256_loadu_ps(&grad_out[b * M * N + i * N + k]);
                                float tmp_b[8] = {0};
                                for (int x = 0; x < 8; ++x) {
                                    size_t idx = j * K + k + x;
                                    if (idx < K * N)
                                        tmp_b[x] = B_T[idx];
                                }
                                __m256 vB_T = _mm256_loadu_ps(tmp_b);
                                vsum_a = _mm256_fmadd_ps(vgrad_out, vB_T, vsum_a);
                            }

                            float buf_a[8];
                            _mm256_storeu_ps(buf_a, vsum_a);
                            float sum_a = buf_a[0] + buf_a[1] + buf_a[2] + buf_a[3] +
                                          buf_a[4] + buf_a[5] + buf_a[6] + buf_a[7];

                            for (; k < k_max; ++k)
                                sum_a += grad_out[b * M * N + i * N + k] * B_T[j * K + k];

                            if (k0 == 0) {
                                if (accumulate)
                                    grad_A[b * M * K + i * K + j] += sum_a;
                                else
                                    grad_A[b * M * K + i * K + j] = sum_a;
                            } else {
                                grad_A[b * M * K + i * K + j] += sum_a;
                            }
                        }
                    }
                }
            }
        }
    }

    // Compute grad_B: batched GEMM reduced over batch, tiled AVX2 kernel (K x N)
    if (grad_B != NULL) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t i = 0; i < K; i += TILE_K) {
            for (size_t j = 0; j < N; j += TILE_N) {
                size_t i_max = (i + TILE_K > K) ? K : i + TILE_K;
                size_t j_max = (j + TILE_N > N) ? N : j + TILE_N;

                for (size_t ii = i; ii < i_max; ++ii) {
                    for (size_t jj = j; jj < j_max; ++jj) {
                        float sum = 0.0f;
                        // Use SIMD for the reduction over batch * M
                        size_t b_m_total = batch * M;
                        size_t k = 0;
                        for (; k + 8 <= b_m_total; k += 8) {
                            float buf_a[8];
                            float buf_g[8];
                            for (int x = 0; x < 8; ++x) {
                                size_t idx = k + x;
                                size_t b_idx = idx / M;
                                size_t m_idx = idx % M;
                                buf_a[x] = A[b_idx * M * K + m_idx * K + ii];
                                buf_g[x] = grad_out[b_idx * M * N + m_idx * N + jj];
                            }
                            __m256 va = _mm256_loadu_ps(buf_a);
                            __m256 vg = _mm256_loadu_ps(buf_g);
                            __m256 vmul = _mm256_mul_ps(va, vg);
                            __m256 vsum = _mm256_hadd_ps(vmul, vmul);
                            float temp[8];
                            _mm256_storeu_ps(temp, vmul);
                            for (int t = 0; t < 8; ++t)
                                sum += temp[t];
                        }
                        // Handle leftover
                        for (; k < b_m_total; ++k) {
                            size_t b_idx = k / M;
                            size_t m_idx = k % M;
                            sum += A[b_idx * M * K + m_idx * K + ii] * grad_out[b_idx * M * N + m_idx * N + jj];
                        }

                        if (accumulate)
                            grad_B[ii * N + jj] += sum;
                        else
                            grad_B[ii * N + jj] = sum;
                    }
                }
            }
        }
    }
}

void tensor_matmul(
    PassMode mode,
    const float* A, const float* B, const float* grad_out,
    float* C_or_A, float* grad_B,
    size_t batch, size_t M, size_t K, size_t N,
    bool accumulate
) {
    if (!A || !B || !C_or_A || (mode == MATMUL_BACKWARD && !grad_out)) {
        fprintf(stderr, "Error: NULL pointer in tensor_matmul\n");
        return;
    }

    if (mode == MATMUL_FORWARD) {
        matmul_forward(A, B, C_or_A, batch, M, K, N);
    } else if (mode == MATMUL_BACKWARD) {
        matmul_backward(A, B, grad_out, C_or_A, grad_B, batch, M, K, N, accumulate);
    }

    if (!matmul_cache_cleanup_registered) {
        atexit(tensor_matmul_free_cache);
        matmul_cache_cleanup_registered = true;
    }
}
