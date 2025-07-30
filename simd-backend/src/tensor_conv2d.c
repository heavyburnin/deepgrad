#include "tensor_conv2d.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>
#include <stdbool.h>

#include "tensor_utils.h"
#include "tensor_matmul.h"

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#ifndef TILE_ROWS
#define TILE_ROWS 2
#endif

static void im2col_rows(const float* restrict input, float* restrict dst,
                        size_t C, size_t H, size_t W,
                        size_t K_h, size_t K_w,
                        size_t pad_h, size_t pad_w,
                        size_t stride_h, size_t stride_w,
                        size_t h_out_start, size_t rows,
                        size_t H_out_total, size_t W_out_total) {
    const size_t patch_stride = C * K_h * K_w;
    #pragma omp parallel for collapse(2)
    for (size_t h_off = 0; h_off < rows; ++h_off) {
        for (size_t w_out = 0; w_out < W_out_total; ++w_out) {
            size_t h_out = h_out_start + h_off;
            float* patch = dst + (h_off * W_out_total + w_out) * patch_stride;
            size_t patch_idx = 0;
            for (size_t c = 0; c < C; ++c) {
                for (size_t kh = 0; kh < K_h; ++kh) {
                    for (size_t kw = 0; kw < K_w; ++kw) {
                        int h_in = (int)(h_out * stride_h + kh) - (int)pad_h;
                        int w_in = (int)(w_out * stride_w + kw) - (int)pad_w;
                        if (patch_idx >= patch_stride) {
                            fprintf(stderr, "im2col_rows: patch_idx=%zu exceeds patch_stride=%zu\n", patch_idx, patch_stride);
                            exit(EXIT_FAILURE);
                        }
                        patch[patch_idx++] = (h_in >= 0 && h_in < (int)H && w_in >= 0 && w_in < (int)W)
                                             ? input[(c * H + h_in) * W + w_in] : 0.0f;
                    }
                }
            }
        }
    }
}

static void col2im_rows(const float* restrict im2col_grad, float* restrict grad_input,
                        size_t C, size_t H, size_t W,
                        size_t K_h, size_t K_w,
                        size_t pad_h, size_t pad_w,
                        size_t stride_h, size_t stride_w,
                        size_t h_out_start, size_t rows,
                        size_t H_out_total, size_t W_out_total) {
    const size_t patch_stride = C * K_h * K_w;
    #pragma omp parallel for collapse(2)
    for (size_t h_off = 0; h_off < rows; ++h_off) {
        for (size_t w_out = 0; w_out < W_out_total; ++w_out) {
            size_t h_out = h_out_start + h_off;
            const float* src = im2col_grad + (h_off * W_out_total + w_out) * patch_stride;
            size_t patch_idx = 0;
            for (size_t c = 0; c < C; ++c) {
                for (size_t kh = 0; kh < K_h; ++kh) {
                    for (size_t kw = 0; kw < K_w; ++kw) {
                        int h_in = (int)(h_out * stride_h + kh) - (int)pad_h;
                        int w_in = (int)(w_out * stride_w + kw) - (int)pad_w;
                        if (patch_idx >= patch_stride) {
                            fprintf(stderr, "col2im_rows: patch_idx=%zu exceeds patch_stride=%zu\n", patch_idx, patch_stride);
                            exit(EXIT_FAILURE);
                        }
                        if (h_in >= 0 && h_in < (int)H && w_in >= 0 && w_in < (int)W) {
                            #pragma omp atomic
                            grad_input[(c * H + h_in) * W + w_in] += src[patch_idx];
                        }
                        ++patch_idx;
                    }
                }
            }
        }
    }
}

void conv2d_forward_gemm(const float* restrict input, const float* restrict weight,
                         const float* restrict bias, float* restrict output,
                         size_t N, size_t C_in, size_t H_in, size_t W_in,
                         size_t C_out, size_t K_h, size_t K_w,
                         size_t stride_h, size_t stride_w,
                         size_t pad_h, size_t pad_w) {
    const size_t H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
    const size_t W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;
    const size_t K = C_in * K_h * K_w;
    const size_t tile_M = TILE_ROWS * W_out;

    if (tile_M * K > (1UL << 28)) {
        fprintf(stderr, "FATAL: tile_M*K too large (%zu floats = %.2f MB)\n",
                tile_M * K, (tile_M * K * sizeof(float)) / (1024.0 * 1024));
        exit(EXIT_FAILURE);
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float* im2col_buf = (float*)_mm_malloc(tile_M * K * sizeof(float), 32);
        if (!im2col_buf) {
            fprintf(stderr, "conv2d_forward_gemm: Thread %d failed to allocate im2col_buf\n", tid);
            exit(EXIT_FAILURE);
        }

        #pragma omp for
        for (size_t n = 0; n < N; ++n) {
            const float* img = input + n * C_in * H_in * W_in;
            float* out_base = output + n * C_out * H_out * W_out;

            for (size_t h_start = 0; h_start < H_out; h_start += TILE_ROWS) {
                const size_t rows = MIN(TILE_ROWS, H_out - h_start);
                const size_t cur_M = rows * W_out;

                im2col_rows(img, im2col_buf, C_in, H_in, W_in, K_h, K_w,
                            pad_h, pad_w, stride_h, stride_w,
                            h_start, rows, H_out, W_out);

                float* out_block = out_base + (h_start * W_out) * C_out;
                matmul_forward(im2col_buf, weight, out_block, 1, cur_M, K, C_out);

                if (bias) {
                    #pragma omp parallel for
                    for (size_t i = 0; i < C_out * cur_M; ++i) {
                        out_block[i] += bias[i % C_out];
                    }
                }
            }
        }
        _mm_free(im2col_buf);
    }
}

void conv2d_backward_gemm(const float* restrict input, const float* restrict weight,
                          const float* restrict grad_out, float* restrict grad_input,
                          float* restrict grad_weight, float* restrict grad_bias,
                          size_t N, size_t C_in, size_t H_in, size_t W_in,
                          size_t C_out, size_t K_h, size_t K_w,
                          size_t stride_h, size_t stride_w,
                          size_t pad_h, size_t pad_w) {

    const size_t H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
    const size_t W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;
    const size_t K = C_in * K_h * K_w;

    if (grad_weight) memset(grad_weight, 0, sizeof(float) * K * C_out);
    if (grad_input) memset(grad_input, 0, sizeof(float) * N * C_in * H_in * W_in);
    if (grad_bias) memset(grad_bias, 0, sizeof(float) * C_out);

    int num_threads = omp_get_max_threads();
    float** thread_weights = calloc(num_threads, sizeof(float*));
    for (int t = 0; t < num_threads; ++t) {
        thread_weights[t] = (float*)_mm_malloc(K * C_out * sizeof(float), 32);
        memset(thread_weights[t], 0, K * C_out * sizeof(float));
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        size_t tile_rows = TILE_ROWS;
        float* im2col_buf = (float*)_mm_malloc(tile_rows * W_out * K * sizeof(float), 32);
        float* grad_im2col_buf = (float*)_mm_malloc(tile_rows * W_out * K * sizeof(float), 32);

        #pragma omp for collapse(2) schedule(static)
        for (size_t n = 0; n < N; ++n) {
            for (size_t h_start = 0; h_start < H_out; h_start += tile_rows) {
                size_t rows = (h_start + tile_rows <= H_out) ? tile_rows : (H_out - h_start);
                size_t cur_M = rows * W_out;

                const float* img = input + n * C_in * H_in * W_in;
                const float* grad_out_n = grad_out + n * C_out * H_out * W_out;
                float* grad_input_n = grad_input ? grad_input + n * C_in * H_in * W_in : NULL;

                im2col_rows(img, im2col_buf, C_in, H_in, W_in, K_h, K_w,
                            pad_h, pad_w, stride_h, stride_w,
                            h_start, rows, H_out, W_out);

                const float* grad_out_block = grad_out_n + (h_start * W_out) * C_out;

                if (grad_weight) {
                    matmul_backward(im2col_buf, grad_out_block, grad_out_block,
                                    NULL, thread_weights[tid],
                                    1, cur_M, K, C_out,
                                    true);
                }

                if (grad_input) {
                    matmul_forward(grad_out_block, weight, grad_im2col_buf,
                                   1, cur_M, C_out, K);

                    col2im_rows(grad_im2col_buf, grad_input_n, C_in, H_in, W_in,
                                K_h, K_w, pad_h, pad_w, stride_h, stride_w,
                                h_start, rows, H_out, W_out);
                }

                if (grad_bias) {
                    for (size_t i = 0; i < rows * W_out; ++i) {
                        const float* gout_row = grad_out_block + i * C_out;
                        #pragma omp simd
                        for (size_t c = 0; c < C_out; ++c) {
                            #pragma omp atomic
                            grad_bias[c] += gout_row[c];
                        }
                    }
                }
            }
        }

        _mm_free(im2col_buf);
        _mm_free(grad_im2col_buf);
    }

    if (grad_weight) {
        for (size_t i = 0; i < K * C_out; ++i) {
            float sum = 0.0f;
            for (int t = 0; t < num_threads; ++t) {
                sum += thread_weights[t][i];
            }
            grad_weight[i] = sum;
        }
    }

    for (int t = 0; t < num_threads; ++t) {
        _mm_free(thread_weights[t]);
    }
    free(thread_weights);
}