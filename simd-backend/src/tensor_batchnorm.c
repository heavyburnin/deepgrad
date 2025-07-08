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

    const size_t N = B * H * W;
    const size_t stride_c = H * W;
    const size_t stride_b = C * stride_c;

    #pragma omp parallel for schedule(dynamic)
    for (size_t c = 0; c < C; ++c) {
        float mean = 0.0f, var = 0.0f;

        for (size_t b = 0; b < B; ++b) {
            const float* base = x + b * stride_b + c * stride_c;
            for (size_t i = 0; i < stride_c; ++i)
                mean += base[i];
        }
        mean /= N;

        for (size_t b = 0; b < B; ++b) {
            const float* base = x + b * stride_b + c * stride_c;
            for (size_t i = 0; i < stride_c; ++i) {
                float diff = base[i] - mean;
                var += diff * diff;
            }
        }
        var = fmaxf(var / N, MIN_VAR);  // clamp for stability

        float std_scalar = 1.0f / sqrtf(var + eps);
        float used_mean = mean;

        if (training) {
            running_mean[c] = momentum * mean + (1.0f - momentum) * running_mean[c];
            running_var[c]  = momentum * var  + (1.0f - momentum) * running_var[c];
        } else {
            used_mean = running_mean[c];
            std_scalar = 1.0f / sqrtf(fmaxf(running_var[c], MIN_VAR) + eps);
        }

        __m256 mean_v  = _mm256_set1_ps(used_mean);
        __m256 std_v   = _mm256_set1_ps(std_scalar);
        __m256 gamma_v = _mm256_set1_ps(gamma[c]);
        __m256 beta_v  = _mm256_set1_ps(beta[c]);

        for (size_t b = 0; b < B; ++b) {
            float* out_base   = out     + b * stride_b + c * stride_c;
            float* xhat_base  = x_hat   + b * stride_b + c * stride_c;
            const float* x_in = x       + b * stride_b + c * stride_c;

            size_t i = 0;
            for (; i + 8 <= stride_c; i += 8) {
                __m256 x_v = _mm256_loadu_ps(x_in + i);
                __m256 xh  = _mm256_mul_ps(_mm256_sub_ps(x_v, mean_v), std_v);
                _mm256_storeu_ps(xhat_base + i, xh);
                __m256 y = _mm256_add_ps(_mm256_mul_ps(gamma_v, xh), beta_v);
                _mm256_storeu_ps(out_base + i, y);
            }

            for (; i < stride_c; ++i) {
                float xh = (x_in[i] - used_mean) * std_scalar;
                xhat_base[i] = xh;
                out_base[i] = gamma[c] * xh + beta[c];
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
    if (!x || !grad_out || !grad_in || !grad_gamma || !grad_beta || !gamma) {
        fprintf(stderr, "Error: batchnorm_backward_f32 received null pointer\n");
        return;
    }

    const size_t m = B * H * W;
    const size_t stride_c = H * W;
    const size_t stride_b = C * stride_c;

    float* local_dgamma = calloc(C, sizeof(float));
    float* local_dbeta  = calloc(C, sizeof(float));

    #pragma omp parallel
    {
        float* thread_dgamma = calloc(C, sizeof(float));
        float* thread_dbeta  = calloc(C, sizeof(float));

        #pragma omp for schedule(dynamic)
        for (size_t c = 0; c < C; ++c) {
            float mean = 0.0f, var = 0.0f;

            for (size_t b = 0; b < B; ++b) {
                const float* x_base = x + b * stride_b + c * stride_c;
                for (size_t i = 0; i < stride_c; ++i)
                    mean += x_base[i];
            }
            mean /= m;

            for (size_t b = 0; b < B; ++b) {
                const float* x_base = x + b * stride_b + c * stride_c;
                for (size_t i = 0; i < stride_c; ++i) {
                    float diff = x_base[i] - mean;
                    var += diff * diff;
                }
            }
            var = fmaxf(var / m, MIN_VAR);
            float std_inv = 1.0f / sqrtf(var + eps);

            // Accumulate gradients
            for (size_t b = 0; b < B; ++b) {
                const float* x_base  = x + b * stride_b + c * stride_c;
                const float* dy_base = grad_out + b * stride_b + c * stride_c;

                for (size_t i = 0; i < stride_c; ++i) {
                    float x_hat = (x_base[i] - mean) * std_inv;
                    thread_dgamma[c] += dy_base[i] * x_hat;
                    thread_dbeta[c]  += dy_base[i];
                }
            }

            for (size_t b = 0; b < B; ++b) {
                const float* x_base  = x + b * stride_b + c * stride_c;
                const float* dy_base = grad_out + b * stride_b + c * stride_c;
                float* dx_base       = grad_in  + b * stride_b + c * stride_c;

                float mean_dy = 0.0f, mean_dy_xhat = 0.0f;

                for (size_t i = 0; i < stride_c; ++i) {
                    float x_hat = (x_base[i] - mean) * std_inv;
                    mean_dy      += dy_base[i];
                    mean_dy_xhat += dy_base[i] * x_hat;
                }

                mean_dy /= m;
                mean_dy_xhat /= m;

                for (size_t i = 0; i < stride_c; ++i) {
                    float x_hat = (x_base[i] - mean) * std_inv;
                    dx_base[i] = gamma[c] * std_inv *
                                 (dy_base[i] - mean_dy - x_hat * mean_dy_xhat);
                }
            }
        }

        // Reduce into global grad_gamma / grad_beta
        #pragma omp critical
        {
            for (size_t c = 0; c < C; ++c) {
                grad_gamma[c] += thread_dgamma[c];
                grad_beta[c]  += thread_dbeta[c];
            }
        }

        free(thread_dgamma);
        free(thread_dbeta);
    }

    free(local_dgamma);
    free(local_dbeta);
}