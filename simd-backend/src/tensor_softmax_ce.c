// tensor_softmax_ce.c

#include "tensor_softmax_ce.h"
#include "tensor_utils.h"

#include <immintrin.h>   // for __m256, AVX intrinsics
#include <stddef.h>      // for size_t
#include <stdbool.h>     // for bool
#include <stdio.h>       // for fprintf, stderr
#include <float.h>       // for FLT_MAX
#include <math.h>        // for expf, logf, fmaxf
#include <omp.h>         // for OpenMP

void tensor_softmax_ce(
    const float* logits,
    const float* labels,
    const float* grad_loss,  // Optional: NULL if not provided
    float* losses,
    float* grad_input,
    float* probs_out,        // Optional: NULL if not needed
    size_t batch,
    size_t class_count
) {
    if (!logits || !labels || !losses || !grad_input) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_softmax_ce_fused\n");
        return;
    }

    if (class_count > MAX_CLASSES) {
        fprintf(stderr, "Error: class_count %zu exceeds MAX_CLASSES (%d)\n", class_count, MAX_CLASSES);
        return;
    }

    const float epsilon = 1e-8f;
    const __m256 v_epsilon = _mm256_set1_ps(epsilon);

    #pragma omp parallel for
    for (size_t b = 0; b < batch; ++b) {
        const float* logits_row = logits + b * class_count;
        const float* labels_row = labels + b * class_count;
        float* probs_row = probs_out ? probs_out + b * class_count : NULL;
        float* grad_row = grad_input + b * class_count;

        // Find max value for numerical stability
        __m256 v_max = _mm256_set1_ps(-FLT_MAX);
        size_t j = 0;
        for (; j + 8 <= class_count; j += 8) {
            __m256 v_logits = _mm256_loadu_ps(logits_row + j);
            v_max = _mm256_max_ps(v_max, v_logits);
        }

        float max_val = hmax256_ps(v_max);
        for (; j < class_count; ++j) {
            if (logits_row[j] > max_val)
                max_val = logits_row[j];
        }

        v_max = _mm256_set1_ps(max_val);

        // Compute exp(logits - max) and sum
        float sum_exp = 0.0f;
        size_t i = 0;
        for (; i + 8 <= class_count; i += 8) {
            __m256 v_logits = _mm256_loadu_ps(logits_row + i);
            __m256 v_shifted = _mm256_sub_ps(v_logits, v_max);
            __m256 v_exp = exp256_ps(v_shifted);
            if (probs_row) _mm256_storeu_ps(probs_row + i, v_exp);
            sum_exp += hsum256_ps(v_exp);
        }

        for (; i < class_count; ++i) {
            float exp_val = expf(logits_row[i] - max_val);
            if (probs_row) probs_row[i] = exp_val;
            sum_exp += exp_val;
        }

        __m256 v_sum_exp = _mm256_set1_ps(sum_exp);
        __m256 v_recip = _mm256_rcp_ps(v_sum_exp);
        v_recip = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(v_sum_exp, v_recip)), v_recip);

        float loss = 0.0f;
        i = 0;
        for (; i + 8 <= class_count; i += 8) {
            __m256 v_probs = _mm256_loadu_ps(probs_row ? probs_row + i : logits_row + i);
            v_probs = _mm256_mul_ps(v_probs, v_recip);  // Normalize

            if (probs_row) _mm256_storeu_ps(probs_row + i, v_probs);

            __m256 v_labels = _mm256_loadu_ps(labels_row + i);
            __m256 v_clamped = _mm256_max_ps(v_probs, v_epsilon);
            __m256 v_log = log256_ps(v_clamped);
            __m256 v_mul = _mm256_mul_ps(v_labels, v_log);
            loss -= hsum256_ps(v_mul);

            __m256 v_diff = _mm256_sub_ps(v_probs, v_labels);
            if (grad_loss) {
                __m256 v_grad_loss = _mm256_set1_ps(grad_loss[b]);
                v_diff = _mm256_mul_ps(v_diff, v_grad_loss);
            }
            _mm256_storeu_ps(grad_row + i, v_diff);
        }

        // Handle tail
        for (; i < class_count; ++i) {
            float prob = probs_row ? probs_row[i] : expf(logits_row[i] - max_val);
            prob /= sum_exp;
            if (probs_row) probs_row[i] = prob;

            float label = labels_row[i];
            float clamped = fmaxf(prob, epsilon);
            loss -= label * logf(clamped);

            float grad_val = prob - label;
            if (grad_loss) grad_val *= grad_loss[b];
            grad_row[i] = grad_val;
        }

        losses[b] = loss;
    }
}