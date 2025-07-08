// tensor_reductions.h

#ifndef TENSOR_REDUCTIONS_H
#define TENSOR_REDUCTIONS_H

#include <stddef.h>  // for size_t
#include <stdbool.h> // for bool
#include <immintrin.h>

// You can define this in the header or externally as a compile-time constant
#define MAX_CLASSES 1024

// Horizontal sum of 8-float AVX vector
float hsum256_ps(__m256 v);

// Horizontal max of 8-float AVX vector
float hmax256_ps(__m256 v);

// Natural log (ln) approximation of 8-float AVX vector
__m256 log256_ps(__m256 x);

// Exponential approximation of 8-float AVX vector
__m256 exp256_ps(__m256 x);

// Compute sum of all elements in a float array
float tensor_sum(const float* input, float* grad_out, size_t len);

// Compute mean of all elements in a float array
float tensor_mean(const float* input, float* grad_out, size_t len);

// Softmax + cross-entropy + gradient (fused)
// - logits: [batch x class_count]
// - labels: one-hot targets, same shape
// - grad_loss: scalar gradient per batch item (or NULL)
// - losses: output loss per batch item
// - grad_input: gradient w.r.t. logits
// - probs_out: optional output for softmax probs (or NULL)
void tensor_softmax_ce_backup(
    const float* logits,
    const int* labels,
    const float* grad_loss,  // Optional: NULL if not provided
    float* losses,
    float* grad_input,
    float* probs_out,        // Optional: NULL if not needed
    size_t batch,
    size_t class_count
);

void tensor_softmax_ce(
    const float* logits,
    const int* labels,
    const float* grad_loss,
    float* losses,
    float* grad_input,
    float* probs_out,
    size_t batch,
    size_t class_count,
    float label_smoothing,
    int use_label_smoothing
);

#endif // TENSOR_REDUCTIONS_H