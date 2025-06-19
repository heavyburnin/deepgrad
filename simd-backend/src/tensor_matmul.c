// tensor_matmul.c

#include "tensor_matmul.h"
#include "tensor_utils.h"   // for get_cached_buffer()
#include <immintrin.h>      // for AVX intrinsics
#include <stdbool.h>        // for bool
#include <stddef.h>         // for size_t
#include <stdio.h>          // for fprintf, stderr
#include <stdlib.h>         // for atexit
#include <mm_malloc.h>      // for _mm_free
#include <omp.h>            // for OpenMP

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

static float* cached_B_T = NULL;
static size_t cached_B_T_size = 0;

static float* cached_grad_out_T = NULL;
static size_t cached_grad_out_T_size = 0;

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
}

void tensor_matmul(
    PassMode mode,
    const float* A, const float* B, const float* grad_out,
    float* C_or_A, float* grad_B,
    size_t batch, size_t M, size_t K, size_t N,
    bool accumulate
) {
    if (!A || !B || !C_or_A || (mode == MATMUL_BACKWARD && !grad_out)) {
        fprintf(stderr, "Error: NULL pointer in tensor_matmul_combined\n");
        return;
    }

    float* B_T = get_cached_buffer(&cached_B_T, &cached_B_T_size, K * N);
    if (!B_T) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return;
    }

    // Transpose B for better memory access
    #pragma omp parallel for collapse(2)
    for (size_t k = 0; k < K; ++k)
        for (size_t n = 0; n < N; ++n)
            B_T[n * K + k] = B[k * N + n];

    if (mode == MATMUL_FORWARD) {
        size_t total_ops = M * K * N;
        if (total_ops < 10000) {
            #pragma omp parallel for
            for (size_t b = 0; b < batch; ++b) {
                const float* A_b = A + b * M * K;
                float* C_b = C_or_A + b * M * N;
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

        #pragma omp parallel for collapse(3)
        for (size_t b = 0; b < batch; ++b) {
            for (size_t i0 = 0; i0 < M; i0 += TILE_M) {
                for (size_t j0 = 0; j0 < N; j0 += TILE_N) {
                    for (size_t k0 = 0; k0 < K; k0 += TILE_K) {
                        const float* A_b = A + b * M * K;
                        float* C_b = C_or_A + b * M * N;

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
    } else if (mode == MATMUL_BACKWARD) {
        float* grad_A = C_or_A;

        // Fused grad_A and grad_B computation
        #pragma omp parallel for collapse(3)
        for (size_t b = 0; b < batch; ++b) {
            for (size_t i0 = 0; i0 < M; i0 += TILE_M) {
                for (size_t j0 = 0; j0 < K; j0 += TILE_K) {
                    for (size_t k0 = 0; k0 < N; k0 += TILE_N) {
                        size_t i_max = (i0 + TILE_M > M) ? M : i0 + TILE_M;
                        size_t j_max = (j0 + TILE_K > K) ? K : j0 + TILE_K;
                        size_t k_max = (k0 + TILE_N > N) ? N : k0 + TILE_N;

                        // Compute grad_A
                        for (size_t i = i0; i < i_max; ++i) {
                            for (size_t j = j0; j < j_max; ++j) {
                                __m256 vsum_a = _mm256_setzero_ps();
                                size_t k = k0;
                                for (; k + 7 < k_max; k += 8) {
                                    __m256 vgrad_out = _mm256_loadu_ps(&grad_out[b * M * N + i * N + k]);
                                    __m256 vB_T = _mm256_loadu_ps(&B_T[k * K + j]);
                                    vsum_a = _mm256_fmadd_ps(vgrad_out, vB_T, vsum_a);
                                }

                                float buf_a[8];
                                _mm256_storeu_ps(buf_a, vsum_a);
                                float sum_a = buf_a[0] + buf_a[1] + buf_a[2] + buf_a[3] +
                                              buf_a[4] + buf_a[5] + buf_a[6] + buf_a[7];

                                for (; k < k_max; ++k)
                                    sum_a += grad_out[b * M * N + i * N + k] * B_T[k * K + j];

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

                        // Compute grad_B
                        for (size_t i = j0; i < j_max; ++i) {  // i iterates over K
                            for (size_t j = k0; j < k_max; ++j) {  // j iterates over N
                                __m256 vsum_b = _mm256_setzero_ps();
                                size_t k = i0;
                                for (; k + 7 < i_max; k += 8) {  // k iterates over M
                                    float buf_a[8], buf_g[8];
                                    for (int x = 0; x < 8; ++x) {
                                        buf_a[x] = A[b * M * K + (k + x) * K + i];
                                        buf_g[x] = grad_out[b * M * N + (k + x) * N + j];
                                    }
                                    __m256 va = _mm256_loadu_ps(buf_a);
                                    __m256 vg = _mm256_loadu_ps(buf_g);
                                    vsum_b = _mm256_fmadd_ps(va, vg, vsum_b);
                                }

                                float buf_b[8];
                                _mm256_storeu_ps(buf_b, vsum_b);
                                float sum_b = buf_b[0] + buf_b[1] + buf_b[2] + buf_b[3] +
                                              buf_b[4] + buf_b[5] + buf_b[6] + buf_b[7];

                                for (; k < i_max; ++k)
                                    sum_b += A[b * M * K + k * K + i] * grad_out[b * M * N + k * N + j];

                                if (i0 == 0) {
                                    if (accumulate)
                                        grad_B[b * K * N + i * N + j] += sum_b;
                                    else
                                        grad_B[b * K * N + i * N + j] = sum_b;
                                } else {
                                    grad_B[b * K * N + i * N + j] += sum_b;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    atexit(tensor_matmul_free_cache);
}

void tensor_matmul_backup(
    PassMode mode,
    const float* A, const float* B, const float* grad_out,
    float* C_or_A, float* grad_B,
    size_t batch, size_t M, size_t K, size_t N,
    bool accumulate
) {
    if (!A || !B || !C_or_A || (mode == MATMUL_BACKWARD && !grad_out)) {
        fprintf(stderr, "Error: NULL pointer in tensor_matmul_combined\n");
        return;
    }

    float* B_T = get_cached_buffer(&cached_B_T, &cached_B_T_size, K * N);
    if (!B_T) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return;
    }

    // Transpose B for better memory access
    #pragma omp parallel for collapse(2)
    for (size_t k = 0; k < K; ++k)
        for (size_t n = 0; n < N; ++n)
            B_T[n * K + k] = B[k * N + n];

    if (mode == MATMUL_FORWARD) {
        size_t total_ops = M * K * N;
        if (total_ops < 10000) {
            #pragma omp parallel for
            for (size_t b = 0; b < batch; ++b) {
                const float* A_b = A + b * M * K;
                float* C_b = C_or_A + b * M * N;
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

        #pragma omp parallel for collapse(3)
        for (size_t b = 0; b < batch; ++b) {
            for (size_t i0 = 0; i0 < M; i0 += TILE_M) {
                for (size_t j0 = 0; j0 < N; j0 += TILE_N) {
                    for (size_t k0 = 0; k0 < K; k0 += TILE_K) {
                        const float* A_b = A + b * M * K;
                        float* C_b = C_or_A + b * M * N;

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
    } else if (mode == MATMUL_BACKWARD) {
        float* grad_A = C_or_A;
        float* grad_out_T = get_cached_buffer(&cached_grad_out_T, &cached_grad_out_T_size, M * N * batch);
        if (!grad_out_T) {
            fprintf(stderr, "Error: Memory allocation failed for grad_out_T\n");
            return;
        }

        // Transpose grad_out: grad_out_T[b * N * M + j * M + i] = grad_out[b * M * N + i * N + j]
        #pragma omp parallel for collapse(3)
        for (size_t b = 0; b < batch; ++b)
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    grad_out_T[b * N * M + j * M + i] = grad_out[b * M * N + i * N + j];


        #pragma omp parallel for collapse(3)
        for (size_t b = 0; b < batch; ++b) {
            for (size_t i0 = 0; i0 < M; i0 += TILE_M) {
                for (size_t j0 = 0; j0 < K; j0 += TILE_K) {
                    for (size_t k0 = 0; k0 < N; k0 += TILE_N) {
                        size_t i_max = (i0 + TILE_M > M) ? M : i0 + TILE_M;
                        size_t j_max = (j0 + TILE_K > K) ? K : j0 + TILE_K;
                        size_t k_max = (k0 + TILE_N > N) ? N : k0 + TILE_N;

                        for (size_t i = i0; i < i_max; ++i) {
                            for (size_t j = j0; j < j_max; ++j) {
                                __m256 vsum = _mm256_setzero_ps();
                                size_t k = k0;
                                for (; k + 7 < k_max; k += 8) {
                                    __m256 vgrad_out = _mm256_loadu_ps(&grad_out[b * M * N + i * N + k]);
                                    __m256 vB_T = _mm256_loadu_ps(&B_T[k * K + j]);
                                    vsum = _mm256_fmadd_ps(vgrad_out, vB_T, vsum);
                                }
                                float buf[8];
                                _mm256_storeu_ps(buf, vsum);
                                float sum = buf[0] + buf[1] + buf[2] + buf[3] +
                                            buf[4] + buf[5] + buf[6] + buf[7];

                                for (; k < k_max; ++k)
                                    sum += grad_out[b * M * N + i * N + k] * B_T[k * K + j];

                                if (k0 == 0) {
                                    if (accumulate)
                                        grad_A[b * M * K + i * K + j] += sum;
                                    else
                                        grad_A[b * M * K + i * K + j] = sum;
                                } else {
                                    grad_A[b * M * K + i * K + j] += sum;
                                }
                            }
                        }
                    }
                }
            }
        }

        #pragma omp parallel for collapse(3)
        // #pragma omp parallel for collapse(3) schedule(dynamic)
        for (size_t b = 0; b < batch; ++b) {
            for (size_t i0 = 0; i0 < K; i0 += TILE_K) {
                for (size_t j0 = 0; j0 < N; j0 += TILE_N) {
                    for (size_t k0 = 0; k0 < M; k0 += TILE_M) {
                        size_t i_max = (i0 + TILE_K > K) ? K : i0 + TILE_K;
                        size_t j_max = (j0 + TILE_N > N) ? N : j0 + TILE_N;
                        size_t k_max = (k0 + TILE_M > M) ? M : k0 + TILE_M;

                        for (size_t i = i0; i < i_max; ++i) {
                            for (size_t j = j0; j < j_max; ++j) {
                                __m256 vsum = _mm256_setzero_ps();
                                size_t k = k0;
                                for (; k + 7 < k_max; k += 8) {
                                    float buf_a[8], buf_g[8];
                                    for (int x = 0; x < 8; ++x) {
                                        buf_a[x] = A[b * M * K + (k + x) * K + i];
                                        buf_g[x] = grad_out_T[b * N * M + j * M + (k + x)];
                                    }
                                    __m256 va = _mm256_loadu_ps(buf_a);
                                    __m256 vg = _mm256_loadu_ps(buf_g);
                                    vsum = _mm256_fmadd_ps(va, vg, vsum);
                                }
                                float buf[8];
                                _mm256_storeu_ps(buf, vsum);
                                float sum = buf[0] + buf[1] + buf[2] + buf[3] +
                                            buf[4] + buf[5] + buf[6] + buf[7];

                                for (; k < k_max; ++k)
                                    sum += A[b * M * K + k * K + i] * grad_out_T[b * N * M + j * M + k];

                                if (k0 == 0) {
                                    if (accumulate)
                                        grad_B[b * K * N + i * N + j] += sum;
                                    else
                                        grad_B[b * K * N + i * N + j] = sum;
                                } else {
                                    grad_B[b * K * N + i * N + j] += sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    atexit(tensor_matmul_free_cache);
}