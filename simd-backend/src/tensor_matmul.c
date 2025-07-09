// tensor_matmul.c

#include "tensor_matmul.h"
#include "tensor_utils.h"   // for get_cached_buffer()
#include <immintrin.h>
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

static float* cached_B_T = NULL;
static size_t cached_B_T_size = 0;
static float* cached_grad_out_T = NULL;
static size_t cached_grad_out_T_size = 0;
static const float* last_B_ptr = NULL;
static size_t last_K = 0;
static size_t last_N = 0;
static bool matmul_cache_cleanup_registered = false;

static inline float horizontal_sum(__m256 vec) {
    float tmp[8];
    _mm256_storeu_ps(tmp, vec);
    return tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
}

void tensor_matmul_free_cache() {
    if (cached_B_T) { _mm_free(cached_B_T); cached_B_T = NULL; cached_B_T_size = 0; }
    if (cached_grad_out_T) { _mm_free(cached_grad_out_T); cached_grad_out_T = NULL; cached_grad_out_T_size = 0; }
    last_B_ptr = NULL; last_K = 0; last_N = 0;
}

void matmul_forward(const float* A, const float* B, float* C, size_t batch, size_t M, size_t K, size_t N) {
    if (!matmul_cache_cleanup_registered) {
        atexit(tensor_matmul_free_cache);
        matmul_cache_cleanup_registered = true;
    }

    float* B_T = get_cached_buffer(&cached_B_T, &cached_B_T_size, K * N);
    if (!B_T) { fprintf(stderr, "Error: Memory allocation failed for B_T\n"); return; }

    if (B != last_B_ptr || K != last_K || N != last_N) {
        last_B_ptr = B; last_K = K; last_N = N;
        #pragma omp parallel for collapse(2)
        for (size_t k = 0; k < K; ++k)
            for (size_t n = 0; n < N; ++n)
                B_T[n * K + k] = B[k * N + n];
    }

    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                __m256 sum_vec = _mm256_setzero_ps();
                size_t k = 0;
                const float* A_b = A + b * M * K;
                const float* B_t = B_T + j * K;
                for (; k + 8 <= K; k += 8) {
                    __m256 a = _mm256_loadu_ps(&A_b[i * K + k]);
                    __m256 b = _mm256_loadu_ps(&B_t[k]);
                    sum_vec = _mm256_fmadd_ps(a, b, sum_vec);
                }
                float sum = horizontal_sum(sum_vec);
                for (; k < K; ++k)
                    sum += A_b[i * K + k] * B_t[k];
                C[b * M * N + i * N + j] = sum;
            }
        }
    }
}

void matmul_backward(const float* A, const float* B, const float* grad_out, float* grad_A, float* grad_B, size_t batch, size_t M, size_t K, size_t N, bool accumulate) {
    if (!matmul_cache_cleanup_registered) {
        atexit(tensor_matmul_free_cache);
        matmul_cache_cleanup_registered = true;
    }

    float* B_T = get_cached_buffer(&cached_B_T, &cached_B_T_size, K * N);
    if (!B_T) { fprintf(stderr, "Error: Memory allocation failed for B_T\n"); return; }

    if (B != last_B_ptr || K != last_K || N != last_N) {
        last_B_ptr = B; last_K = K; last_N = N;
        #pragma omp parallel for collapse(2)
        for (size_t k = 0; k < K; ++k)
            for (size_t n = 0; n < N; ++n)
                B_T[n * K + k] = B[k * N + n];
    }

    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                __m256 sum_vec = _mm256_setzero_ps();
                const float* gout = grad_out + b * M * N + i * N;
                const float* B_ptr = B + k * N;
                size_t j = 0;
                for (; j + 8 <= N; j += 8) {
                    __m256 go = _mm256_loadu_ps(&gout[j]);
                    __m256 bt = _mm256_loadu_ps(&B_ptr[j]);
                    sum_vec = _mm256_fmadd_ps(go, bt, sum_vec);
                }
                float sum = horizontal_sum(sum_vec);
                for (; j < N; ++j)
                    sum += gout[j] * B_ptr[j];
                size_t idx = b * M * K + i * K + k;
                if (accumulate) grad_A[idx] += sum; else grad_A[idx] = sum;
            }
        }
    }

    if (grad_B) {
        #pragma omp parallel for collapse(2)
        for (size_t k = 0; k < K; ++k) {
            for (size_t n = 0; n < N; ++n) {
                __m256 sum_vec = _mm256_setzero_ps();
                for (size_t b = 0; b < batch; ++b) {
                    for (size_t m = 0; m < M; ++m) {
                        float a = A[b * M * K + m * K + k];
                        float g = grad_out[b * M * N + m * N + n];
                        __m256 va = _mm256_broadcast_ss(&a);
                        __m256 vg = _mm256_broadcast_ss(&g);
                        sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(va, vg));
                    }
                }
                float sum = horizontal_sum(sum_vec);
                if (accumulate) grad_B[k * N + n] += sum; else grad_B[k * N + n] = sum;
            }
        }
    }
}

