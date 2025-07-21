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
    printf("conv2d_backward_gemm: input=%p, weight=%p, grad_out=%p, grad_input=%p, grad_weight=%p, grad_bias=%p\n",
           input, weight, grad_out, grad_input, grad_weight, grad_bias);
    printf("conv2d_backward_gemm: N=%zu, C_in=%zu, H_in=%zu, W_in=%zu, C_out=%zu, K_h=%zu, K_w=%zu, stride_h=%zu, stride_w=%zu, pad_h=%zu, pad_w=%zu\n",
           N, C_in, H_in, W_in, C_out, K_h, K_w, stride_h, stride_w, pad_h, pad_w);

    const size_t H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
    const size_t W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;
    const size_t K = C_in * K_h * K_w;
    const size_t tile_M = TILE_ROWS * W_out;

    printf("conv2d_backward_gemm: H_out=%zu, W_out=%zu, K=%zu, tile_M=%zu\n",
           H_out, W_out, K, tile_M);

    if (tile_M * K > (1UL << 28)) {
        fprintf(stderr, "FATAL: tile_M*K too large (%zu floats = %.2f MB)\n",
                tile_M * K, (tile_M * K * sizeof(float)) / (1024.0 * 1024));
        exit(EXIT_FAILURE);
    }

    // Debug grad_out sum
    float grad_out_sum = 0.0f;
    for (size_t i = 0; i < N * C_out * H_out * W_out; ++i) grad_out_sum += grad_out[i];
    printf("conv2d_backward_gemm: grad_out sum=%f\n", grad_out_sum);

    if (grad_weight) memset(grad_weight, 0, sizeof(float) * K * C_out);
    if (grad_input) memset(grad_input, 0, sizeof(float) * N * C_in * H_in * W_in);
    if (grad_bias) memset(grad_bias, 0, sizeof(float) * C_out);

    int num_threads = omp_get_max_threads();
    float** thread_weights = calloc(num_threads, sizeof(float*));
    if (!thread_weights) {
        fprintf(stderr, "conv2d_backward_gemm: Failed to allocate thread_weights\n");
        exit(EXIT_FAILURE);
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        thread_weights[tid] = (float*)_mm_malloc(K * C_out * sizeof(float), 32);
        if (!thread_weights[tid]) {
            fprintf(stderr, "conv2d_backward_gemm: Thread %d failed to allocate local_grad_weight\n", tid);
            exit(EXIT_FAILURE);
        }
        memset(thread_weights[tid], 0, K * C_out * sizeof(float));
        float* local_grad_weight = thread_weights[tid];
        printf("conv2d_backward_gemm: Thread %d, local_grad_weight=%p\n", tid, local_grad_weight);

        float* im2col_buf = (float*)_mm_malloc(tile_M * K * sizeof(float), 32);
        float* grad_im2col_buf = (float*)_mm_malloc(tile_M * K * sizeof(float), 32);
        if (!im2col_buf || !grad_im2col_buf) {
            fprintf(stderr, "conv2d_backward_gemm: Thread %d failed to allocate im2col_buf=%p or grad_im2col_buf=%p\n",
                    tid, im2col_buf, grad_im2col_buf);
            exit(EXIT_FAILURE);
        }
        printf("conv2d_backward_gemm: Thread %d, im2col_buf=%p, grad_im2col_buf=%p\n",
               tid, im2col_buf, grad_im2col_buf);

        #pragma omp for
        for (size_t n = 0; n < N; ++n) {
            const float* img = input + n * C_in * H_in * W_in;
            const float* grad_out_n = grad_out + n * C_out * H_out * W_out;
            float* grad_input_n = grad_input ? grad_input + n * C_in * H_in * W_in : NULL;
            printf("conv2d_backward_gemm: Thread %d, n=%zu, img=%p, grad_out_n=%p, grad_input_n=%p\n",
                   tid, n, img, grad_out_n, grad_input_n);

            for (size_t h_start = 0; h_start < H_out; h_start += TILE_ROWS) {
                size_t rows = MIN(TILE_ROWS, H_out - h_start);
                size_t cur_M = rows * W_out;
                printf("conv2d_backward_gemm: Thread %d, h_start=%zu, rows=%zu, cur_M=%zu\n",
                       tid, h_start, rows, cur_M);

                // Debug buffer contents before im2col_rows
                float sum_im2col = 0.0f;
                for (size_t i = 0; i < tile_M * K; ++i) sum_im2col += im2col_buf[i];
                printf("conv2d_backward_gemm: Thread %d, im2col_buf sum before=%f\n", tid, sum_im2col);

                im2col_rows(img, im2col_buf, C_in, H_in, W_in, K_h, K_w,
                            pad_h, pad_w, stride_h, stride_w,
                            h_start, rows, H_out, W_out);
                printf("conv2d_backward_gemm: Thread %d, im2col_rows completed, im2col_buf=%p\n", tid, im2col_buf);

                // Debug buffer contents after im2col_rows
                sum_im2col = 0.0f;
                for (size_t i = 0; i < tile_M * K; ++i) sum_im2col += im2col_buf[i];
                printf("conv2d_backward_gemm: Thread %d, im2col_buf sum after=%f\n", tid, sum_im2col);

                const float* grad_out_block = grad_out_n + (h_start * W_out) * C_out;
                // Check alignment
                if ((uintptr_t)grad_out_block % 32 != 0) {
                    fprintf(stderr, "conv2d_backward_gemm: Thread %d, grad_out_block=%p not 32-byte aligned\n", tid, grad_out_block);
                    exit(EXIT_FAILURE);
                }
                if (grad_input_n && (uintptr_t)grad_input_n % 32 != 0) {
                    fprintf(stderr, "conv2d_backward_gemm: Thread %d, grad_input_n=%p not 32-byte aligned\n", tid, grad_input_n);
                    exit(EXIT_FAILURE);
                }

                float sum_grad_out = 0.0f;
                for (size_t i = 0; i < cur_M * C_out; ++i) sum_grad_out += grad_out_block[i];
                printf("conv2d_backward_gemm: Thread %d, grad_out_block=%p, shape=(%zu, %zu), sum=%f\n",
                       tid, grad_out_block, cur_M, C_out, sum_grad_out);

                if (grad_weight) {
                    printf("conv2d_backward_gemm: Thread %d, calling matmul_backward for grad_weight, A=%p, B=%p, grad_out=%p, grad_A=%p, grad_B=%p, M=%zu, K=%zu, N=%zu\n",
                           tid, im2col_buf, weight, grad_out_block, local_grad_weight, NULL, K, cur_M, C_out);
                    matmul_backward(im2col_buf, weight, grad_out_block, local_grad_weight, NULL,
                                    1, K, cur_M, C_out, true);

                    // Debug local_grad_weight
                    float sum_grad_weight = 0.0f;
                    for (size_t i = 0; i < K * C_out; ++i) sum_grad_weight += local_grad_weight[i];
                    printf("conv2d_backward_gemm: Thread %d, local_grad_weight sum=%f\n", tid, sum_grad_weight);
                }

                if (grad_input) {
                    printf("conv2d_backward_gemm: Thread %d, calling matmul_forward for grad_input, A=%p, B=%p, out=%p\n",
                           tid, grad_out_block, weight, grad_im2col_buf);
                    matmul_forward(grad_out_block, weight, grad_im2col_buf, 1, cur_M, C_out, K);

                    // Debug grad_im2col_buf
                    float sum_grad_im2col = 0.0f;
                    for (size_t i = 0; i < cur_M * K; ++i) sum_grad_im2col += grad_im2col_buf[i];
                    printf("conv2d_backward_gemm: Thread %d, grad_im2col_buf sum=%f\n", tid, sum_grad_im2col);

                    col2im_rows(grad_im2col_buf, grad_input_n, C_in, H_in, W_in,
                                K_h, K_w, pad_h, pad_w, stride_h, stride_w,
                                h_start, rows, H_out, W_out);
                    printf("conv2d_backward_gemm: Thread %d, col2im_rows completed\n", tid);

                    // Debug grad_input_n
                    float sum_grad_input = 0.0f;
                    for (size_t i = 0; i < C_in * H_in * W_in; ++i) sum_grad_input += grad_input_n[i];
                    printf("conv2d_backward_gemm: Thread %d, grad_input_n sum=%f\n", tid, sum_grad_input);
                }
            }
        }
        _mm_free(im2col_buf);
        _mm_free(grad_im2col_buf);
        _mm_free(thread_weights[tid]);
        thread_weights[tid] = NULL; // Prevent double-free
    }

    if (grad_weight) {
        #pragma omp parallel for
        for (size_t i = 0; i < K * C_out; ++i) {
            for (int t = 0; t < num_threads; ++t) {
                if (thread_weights[t]) {
                    grad_weight[i] += thread_weights[t][i];
                }
            }
        }
    }
    free(thread_weights);
}
