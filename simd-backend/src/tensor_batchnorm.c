#include "tensor_batchnorm.h"
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#define MIN_VAR 1e-5f

static inline __m256 rsqrt_newton(__m256 x) {
    __m256 y = _mm256_rsqrt_ps(x);
    const __m256 three = _mm256_set1_ps(3.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    y = _mm256_mul_ps(y, _mm256_sub_ps(three, _mm256_mul_ps(_mm256_mul_ps(x, y), y)));
    return _mm256_mul_ps(y, half);
}

void batchnorm_forward_f32(
    const float* restrict x, float* restrict out, float* restrict x_hat,
    const float* restrict gamma, const float* restrict beta,
    float* restrict running_mean, float* restrict running_var,
    size_t B, size_t C, size_t H, size_t W,
    float eps, float momentum, bool training
) {
    if (!x || !out || !x_hat || !gamma || !beta || !running_mean || !running_var) {
        fprintf(stderr, "Error: batchnorm_forward_f32 received null pointer\n");
        return;
    }

    const size_t spatial = H * W;
    const size_t N = B * spatial;

    #pragma omp parallel for schedule(static)
    for (size_t c = 0; c < C; ++c) {
        float mean = 0.0f, var = 0.0f;

        // Compute mean
        #pragma omp simd reduction(+:mean)
        for (size_t i = 0; i < B * spatial; ++i)
            mean += x[c * spatial + i + (i / spatial) * (C - 1) * spatial];

        mean /= N;

        // Compute variance
        #pragma omp simd reduction(+:var)
        for (size_t i = 0; i < B * spatial; ++i) {
            float val = x[c * spatial + i + (i / spatial) * (C - 1) * spatial];
            float diff = val - mean;
            var += diff * diff;
        }
        var = fmaxf(var / N, MIN_VAR);
        float std_inv = 1.0f / sqrtf(var + eps);
        float used_mean = mean;

        if (training) {
            running_mean[c] = momentum * mean + (1.0f - momentum) * running_mean[c];
            running_var[c]  = momentum * var  + (1.0f - momentum) * running_var[c];
        } else {
            used_mean = running_mean[c];
            std_inv = 1.0f / sqrtf(fmaxf(running_var[c], MIN_VAR) + eps);
        }

        const __m256 mean_v  = _mm256_set1_ps(used_mean);
        const __m256 std_v   = _mm256_set1_ps(std_inv);
        const __m256 gamma_v = _mm256_set1_ps(gamma[c]);
        const __m256 beta_v  = _mm256_set1_ps(beta[c]);

        for (size_t b = 0; b < B; ++b) {
            const size_t offset = b * C * spatial + c * spatial;
            const float* x_ptr = x + offset;
            float* xh_ptr = x_hat + offset;
            float* out_ptr = out + offset;

            size_t i = 0;
            for (; i + 8 <= spatial; i += 8) {
                __m256 x_v = _mm256_loadu_ps(x_ptr + i);
                __m256 xh  = _mm256_mul_ps(_mm256_sub_ps(x_v, mean_v), std_v);
                _mm256_storeu_ps(xh_ptr + i, xh);
                __m256 y = _mm256_add_ps(_mm256_mul_ps(gamma_v, xh), beta_v);
                _mm256_storeu_ps(out_ptr + i, y);
            }

            for (; i < spatial; ++i) {
                float xh = (x_ptr[i] - used_mean) * std_inv;
                xh_ptr[i] = xh;
                out_ptr[i] = gamma[c] * xh + beta[c];
            }
        }
    }
}

void batchnorm_backward_f32(
    const float* restrict x, const float* restrict grad_out,
    float* restrict grad_in, float* restrict grad_gamma, float* restrict grad_beta,
    const float* restrict gamma,
    size_t B, size_t C, size_t H, size_t W,
    float eps
) {
    const size_t spatial = H * W;
    const size_t N = B * spatial;

    #pragma omp parallel for schedule(static)
    for (size_t c = 0; c < C; ++c) {
        float mean = 0.0f, var = 0.0f;

        // Compute mean
        #pragma omp simd reduction(+:mean)
        for (size_t b = 0; b < B; ++b) {
            const float* x_ptr = x + b * C * spatial + c * spatial;
            for (size_t i = 0; i < spatial; ++i)
                mean += x_ptr[i];
        }
        mean /= N;

        // Compute variance
        #pragma omp simd reduction(+:var)
        for (size_t b = 0; b < B; ++b) {
            const float* x_ptr = x + b * C * spatial + c * spatial;
            for (size_t i = 0; i < spatial; ++i) {
                float diff = x_ptr[i] - mean;
                var += diff * diff;
            }
        }
        var = fmaxf(var / N, MIN_VAR);
        float std_inv = 1.0f / sqrtf(var + eps);

        float dgamma = 0.0f, dbeta = 0.0f;
        float mean_dy = 0.0f, mean_dy_xhat = 0.0f;

        // First pass: compute dgamma, dbeta, and intermediates
        for (size_t b = 0; b < B; ++b) {
            const float* x_ptr  = x + b * C * spatial + c * spatial;
            const float* dy_ptr = grad_out + b * C * spatial + c * spatial;
            for (size_t i = 0; i < spatial; ++i) {
                float x_hat = (x_ptr[i] - mean) * std_inv;
                dgamma += dy_ptr[i] * x_hat;
                dbeta  += dy_ptr[i];
                mean_dy += dy_ptr[i];
                mean_dy_xhat += dy_ptr[i] * x_hat;
            }
        }

        mean_dy /= N;
        mean_dy_xhat /= N;

        // Second pass: compute dx
        for (size_t b = 0; b < B; ++b) {
            const float* x_ptr  = x + b * C * spatial + c * spatial;
            const float* dy_ptr = grad_out + b * C * spatial + c * spatial;
            float* dx_ptr = grad_in + b * C * spatial + c * spatial;

            for (size_t i = 0; i < spatial; ++i) {
                float x_hat = (x_ptr[i] - mean) * std_inv;
                dx_ptr[i] = gamma[c] * std_inv *
                            (dy_ptr[i] - mean_dy - x_hat * mean_dy_xhat);
            }
        }

        #pragma omp atomic
        grad_gamma[c] += dgamma;
        #pragma omp atomic
        grad_beta[c] += dbeta;
    }
}
