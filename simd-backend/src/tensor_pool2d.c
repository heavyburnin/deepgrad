#include "tensor_pool2d.h"
#include <stddef.h>
#include <immintrin.h>
#include <omp.h>
#include <string.h>
#include <float.h>

void avgpool2d_forward(const float* input, float* output,
                       size_t N, size_t C, size_t H, size_t W,
                       size_t kernel_h, size_t kernel_w,
                       size_t stride_h, size_t stride_w) {
    size_t H_out = (H - kernel_h + stride_h) / stride_h;
    size_t W_out = (W - kernel_w + stride_w) / stride_w;

    #pragma omp parallel for collapse(2)
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t h_out = 0; h_out < H_out; ++h_out) {
                for (size_t w_out = 0; w_out < W_out; ++w_out) {
                    float sum = 0.0f;
                    int count = 0;

                    for (size_t kh = 0; kh < kernel_h; ++kh) {
                        for (size_t kw = 0; kw < kernel_w; ++kw) {
                            size_t h = h_out * stride_h + kh;
                            size_t w = w_out * stride_w + kw;
                            if (h < H && w < W) {
                                sum += input[((n * C + c) * H + h) * W + w];
                                count++;
                            }
                        }
                    }

                    output[((n * C + c) * H_out + h_out) * W_out + w_out] =
                        count > 0 ? sum / count : 0.0f;
                }
            }
        }
    }
}

void avgpool2d_backward(const float* grad_out, float* grad_input,
                        size_t N, size_t C, size_t H, size_t W,
                        size_t kernel_h, size_t kernel_w,
                        size_t stride_h, size_t stride_w) {
    size_t H_out = (H - kernel_h + stride_h) / stride_h;
    size_t W_out = (W - kernel_w + stride_w) / stride_w;

    memset(grad_input, 0, sizeof(float) * N * C * H * W);

    #pragma omp parallel for collapse(2)
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t h_out = 0; h_out < H_out; ++h_out) {
                for (size_t w_out = 0; w_out < W_out; ++w_out) {
                    float grad_val = grad_out[((n * C + c) * H_out + h_out) * W_out + w_out];
                    int count = 0;

                    // First count how many valid positions there are
                    for (size_t kh = 0; kh < kernel_h; ++kh) {
                        for (size_t kw = 0; kw < kernel_w; ++kw) {
                            size_t h = h_out * stride_h + kh;
                            size_t w = w_out * stride_w + kw;
                            if (h < H && w < W) {
                                count++;
                            }
                        }
                    }

                    float grad = count > 0 ? grad_val / count : 0.0f;

                    for (size_t kh = 0; kh < kernel_h; ++kh) {
                        for (size_t kw = 0; kw < kernel_w; ++kw) {
                            size_t h = h_out * stride_h + kh;
                            size_t w = w_out * stride_w + kw;
                            if (h < H && w < W) {
                                #pragma omp atomic
                                grad_input[((n * C + c) * H + h) * W + w] += grad;
                            }
                        }
                    }
                }
            }
        }
    }
}

void maxpool2d_forward(const float* input, float* output,
                       size_t N, size_t C, size_t H, size_t W,
                       size_t kernel_h, size_t kernel_w,
                       size_t stride_h, size_t stride_w) {
    size_t H_out = (H - kernel_h + stride_h) / stride_h;
    size_t W_out = (W - kernel_w + stride_w) / stride_w;

    #pragma omp parallel for collapse(2)
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t h_out = 0; h_out < H_out; ++h_out) {
                for (size_t w_out = 0; w_out < W_out; ++w_out) {
                    float max_val = -FLT_MAX;

                    for (size_t kh = 0; kh < kernel_h; ++kh) {
                        for (size_t kw = 0; kw < kernel_w; ++kw) {
                            size_t h = h_out * stride_h + kh;
                            size_t w = w_out * stride_w + kw;
                            if (h < H && w < W) {
                                float val = input[((n * C + c) * H + h) * W + w];
                                if (val > max_val) max_val = val;
                            }
                        }
                    }

                    output[((n * C + c) * H_out + h_out) * W_out + w_out] = max_val;
                }
            }
        }
    }
}

void maxpool2d_backward(const float* input, const float* grad_out, float* grad_input,
                        size_t N, size_t C, size_t H, size_t W,
                        size_t kernel_h, size_t kernel_w,
                        size_t stride_h, size_t stride_w) {
    size_t H_out = (H - kernel_h + stride_h) / stride_h;
    size_t W_out = (W - kernel_w + stride_w) / stride_w;

    memset(grad_input, 0, sizeof(float) * N * C * H * W);

    #pragma omp parallel for collapse(2)
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t h_out = 0; h_out < H_out; ++h_out) {
                for (size_t w_out = 0; w_out < W_out; ++w_out) {
                    float max_val = -FLT_MAX;
                    size_t max_h = 0, max_w = 0;

                    for (size_t kh = 0; kh < kernel_h; ++kh) {
                        for (size_t kw = 0; kw < kernel_w; ++kw) {
                            size_t h = h_out * stride_h + kh;
                            size_t w = w_out * stride_w + kw;
                            if (h < H && w < W) {
                                float val = input[((n * C + c) * H + h) * W + w];
                                if (val > max_val) {
                                    max_val = val;
                                    max_h = h;
                                    max_w = w;
                                }
                            }
                        }
                    }

                    float grad = grad_out[((n * C + c) * H_out + h_out) * W_out + w_out];
                    #pragma omp atomic
                    grad_input[((n * C + c) * H + max_h) * W + max_w] += grad;
                }
            }
        }
    }
}
