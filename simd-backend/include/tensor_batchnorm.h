// tensor_batchnorm.h
#pragma once
#include <stddef.h>
#include <stdbool.h>

void batchnorm_forward_f32(
    const float* restrict x, float* restrict out, float* restrict x_hat,
    const float* restrict gamma, const float* restrict beta,
    float* restrict running_mean, float* restrict running_var,
    size_t B, size_t C, size_t H, size_t W,
    float eps, float momentum, bool training
);

void batchnorm_backward_f32(
    const float* restrict x_hat, const float* restrict grad_out,
    float* restrict grad_in, float* restrict grad_gamma, float* restrict grad_beta,
    const float* restrict gamma,
    size_t B, size_t C, size_t H, size_t W,
    float eps
);
