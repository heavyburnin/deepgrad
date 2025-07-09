#include "tensor_conv2d.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>
#include <stdbool.h>

#include "tensor_utils.h"
#include "tensor_matmul.h"

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

static float* cached_im2col = NULL;
static size_t cached_im2col_size = 0;

static float* cached_grad_im2col = NULL;
static size_t cached_grad_im2col_size = 0;

static inline float* aligned_malloc(size_t size) {
    return (float*)_mm_malloc(sizeof(float) * size, 32);
}

static inline void aligned_free(float* ptr) {
    _mm_free(ptr);
}

void im2col_single(const float* input, float* dst,
                   size_t C, size_t H, size_t W,
                   size_t K_h, size_t K_w,
                   size_t pad_h, size_t pad_w,
                   size_t stride_h, size_t stride_w,
                   size_t H_out, size_t W_out) {
    int H_int = (int)H;
    int W_int = (int)W;

    #pragma omp parallel for collapse(2)
    for (size_t h_out = 0; h_out < H_out; ++h_out) {
        for (size_t w_out = 0; w_out < W_out; ++w_out) {
            size_t out_row = h_out * W_out + w_out;
            float* patch = dst + out_row * C * K_h * K_w;

            size_t patch_idx = 0;
            for (size_t c = 0; c < C; ++c) {
                for (size_t kh = 0; kh < K_h; ++kh) {
                    for (size_t kw = 0; kw < K_w; ++kw) {
                        int h_in = (int)(h_out * stride_h + kh) - (int)pad_h;
                        int w_in = (int)(w_out * stride_w + kw) - (int)pad_w;
                        float val = 0.0f;
                        if (h_in >= 0 && h_in < H_int && w_in >= 0 && w_in < W_int)
                            val = input[(c * H + (size_t)h_in) * W + (size_t)w_in];
                        patch[patch_idx++] = val;
                    }
                }
            }
        }
    }
}

void col2im(const float* im2col_grad, float* grad_input,
            size_t C, size_t H, size_t W,
            size_t K_h, size_t K_w,
            size_t pad_h, size_t pad_w,
            size_t stride_h, size_t stride_w,
            size_t H_out, size_t W_out) {

    memset(grad_input, 0, sizeof(float) * C * H * W);

    int H_int = (int)H;
    int W_int = (int)W;

    #pragma omp parallel for collapse(2)
    for (size_t h_out = 0; h_out < H_out; ++h_out) {
        for (size_t w_out = 0; w_out < W_out; ++w_out) {
            size_t out_row = h_out * W_out + w_out;
            const float* src = im2col_grad + out_row * C * K_h * K_w;

            size_t patch_idx = 0;
            for (size_t c = 0; c < C; ++c) {
                for (size_t kh = 0; kh < K_h; ++kh) {
                    for (size_t kw = 0; kw < K_w; ++kw) {
                        int h_in = (int)(h_out * stride_h + kh) - (int)pad_h;
                        int w_in = (int)(w_out * stride_w + kw) - (int)pad_w;
                        if (h_in >= 0 && h_in < H_int && w_in >= 0 && w_in < W_int) {
                            #pragma omp atomic
                            grad_input[(c * H + (size_t)h_in) * W + (size_t)w_in] += src[patch_idx];
                        }
                        patch_idx++;
                    }
                }
            }
        }
    }
}

void conv2d_forward_gemm(const float* input, const float* weight, const float* bias, float* output,
                         size_t N, size_t C_in, size_t H_in, size_t W_in,
                         size_t C_out, size_t K_h, size_t K_w,
                         size_t stride_h, size_t stride_w,
                         size_t pad_h, size_t pad_w) {
    size_t H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
    size_t W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;
    size_t M = H_out * W_out;
    size_t K = C_in * K_h * K_w;
    size_t im2col_size = N * M * K;

    float* im2col_buf = get_cached_buffer(&cached_im2col, &cached_im2col_size, im2col_size);
    if (!im2col_buf) {
        fprintf(stderr, "Error: im2col buffer allocation failed\n");
        return;
    }

    #pragma omp parallel for
    for (size_t n = 0; n < N; ++n) {
        const float* img = input + n * C_in * H_in * W_in;
        float* dst = im2col_buf + n * M * K;
        im2col_single(img, dst, C_in, H_in, W_in, K_h, K_w,
                      pad_h, pad_w, stride_h, stride_w, H_out, W_out);
    }

    matmul_forward(im2col_buf, weight, output, N, M, K, C_out);

    if (bias) {
        #pragma omp parallel for
        for (size_t i = 0; i < N * C_out * H_out * W_out; ++i) {
            size_t c = (i / (H_out * W_out)) % C_out;
            output[i] += bias[c];
        }
    }
}

void conv2d_backward_gemm(const float* input, const float* weight, const float* grad_out,
                          float* grad_input, float* grad_weight, float* grad_bias,
                          size_t N, size_t C_in, size_t H_in, size_t W_in,
                          size_t C_out, size_t K_h, size_t K_w,
                          size_t stride_h, size_t stride_w,
                          size_t pad_h, size_t pad_w) {

    size_t H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
    size_t W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;
    size_t M = H_out * W_out;
    size_t K = C_in * K_h * K_w;
    size_t total_size = N * M * K;

    float* im2col_buf = get_cached_buffer(&cached_im2col, &cached_im2col_size, total_size);
    float* grad_im2col = get_cached_buffer(&cached_grad_im2col, &cached_grad_im2col_size, total_size);

    if (!im2col_buf || !grad_im2col) {
        fprintf(stderr, "Error: buffer allocation failed\n");
        return;
    }

    memset(grad_weight, 0, sizeof(float) * K * C_out);
    memset(grad_input, 0, sizeof(float) * N * C_in * H_in * W_in);

    #pragma omp parallel for
    for (size_t n = 0; n < N; ++n) {
        const float* img = input + n * C_in * H_in * W_in;
        float* dst = im2col_buf + n * M * K;
        im2col_single(img, dst, C_in, H_in, W_in, K_h, K_w,
                      pad_h, pad_w, stride_h, stride_w, H_out, W_out);
    }

    matmul_backward(im2col_buf, grad_out, NULL, grad_weight, NULL,
                N, K, M, C_out, true);

    matmul_forward(grad_out, weight, grad_im2col, N, M, C_out, K);

    #pragma omp parallel for
    for (size_t n = 0; n < N; ++n) {
        float* dst = grad_input + n * C_in * H_in * W_in;
        const float* src = grad_im2col + n * M * K;
        col2im(src, dst, C_in, H_in, W_in,
               K_h, K_w, pad_h, pad_w,
               stride_h, stride_w, H_out, W_out);
    }

    if (grad_bias) {
        // Safe and portable across OpenMP versions
        #pragma omp parallel
        {
            float* local_sum = (float*)calloc(C_out, sizeof(float));
            #pragma omp for
            for (size_t n = 0; n < N; ++n) {
                for (size_t c = 0; c < C_out; ++c) {
                    for (size_t h = 0; h < H_out; ++h) {
                        for (size_t w = 0; w < W_out; ++w) {
                            size_t idx = ((n * C_out + c) * H_out + h) * W_out + w;
                            local_sum[c] += grad_out[idx];
                        }
                    }
                }
            }
            #pragma omp critical
            {
                for (size_t c = 0; c < C_out; ++c) {
                    grad_bias[c] += local_sum[c];
                }
            }
            free(local_sum);
        }
    }
}