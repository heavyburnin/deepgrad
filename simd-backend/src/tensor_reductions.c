// tensor_reductions.c

#include "tensor_reductions.h"
#include "tensor_utils.h"   // For hsum256_ps or other SIMD helpers
#include <immintrin.h>      // For AVX intrinsics
#include <stddef.h>         // For 
#include <stdio.h>          // Error logging
#include <omp.h>            // For OpenMP multithreading
#include <stdbool.h>     // for bool
#include <float.h>       // for FLT_MAX
#include <math.h>        // for expf, logf, fmaxf

float hsum256_ps(__m256 v) {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_movehdup_ps(sum128));
    return _mm_cvtss_f32(sum128);
}

float hmax256_ps(__m256 v) {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 max128 = _mm_max_ps(low, high);
    max128 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    max128 = _mm_max_ss(max128, _mm_movehdup_ps(max128));
    return _mm_cvtss_f32(max128);
}

__m256 exp256_ps(__m256 x) {
    const __m256 ln2 = _mm256_set1_ps(0.69314718056f);
    const __m256 inv_ln2 = _mm256_set1_ps(1.44269504089f);  // 1/ln(2)
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256i bias = _mm256_set1_epi32(127);

    // clamp x to avoid overflow
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.336544f)), _mm256_set1_ps(88.722839f));

    // n = floor(x / ln2 + 0.5)
    __m256 fx = _mm256_fmadd_ps(x, inv_ln2, _mm256_set1_ps(0.5f));
    __m256i emm0 = _mm256_cvttps_epi32(fx);
    fx = _mm256_cvtepi32_ps(emm0);

    // r = x - n * ln2
    __m256 r = _mm256_fnmadd_ps(fx, ln2, x);

    // polynomial approximation for exp(r)
    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(5.0000001201E-1f));
    y = _mm256_fmadd_ps(y, r, one);

    // 2^n
    __m256i pow2n = _mm256_slli_epi32(_mm256_add_epi32(emm0, bias), 23);
    __m256 result = _mm256_mul_ps(y, _mm256_castsi256_ps(pow2n));
    return result;
}

float tensor_sum(const float* input, float* grad_out, size_t len) {
    if (!input) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_sum\n");
        return 0.0f;
    }

    float total_sum = 0.0f;
    size_t vec_end = len - (len % 8);

    #pragma omp parallel reduction(+:total_sum)
    {
        __m256 vsum = _mm256_setzero_ps();

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < vec_end; i += 8) {
            __m256 v = _mm256_loadu_ps(input + i);
            vsum = _mm256_add_ps(vsum, v);

            // Optional: write gradient directly
            if (grad_out) {
                _mm256_storeu_ps(grad_out + i, _mm256_set1_ps(1.0f));  // d(sum)/dx = 1
            }
        }

        float partial = hsum256_ps(vsum);
        total_sum += partial;
    }

    // Handle tail
    for (size_t i = vec_end; i < len; i++) {
        total_sum += input[i];
        if (grad_out) grad_out[i] = 1.0f;
    }

    return total_sum;
}

float tensor_mean(const float* input, float* grad_out, size_t len) {
    if (!input) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_mean\n");
        return 0.0f;
    }

    if (len == 0) {
        fprintf(stderr, "Error: Division by zero in tensor_mean\n");
        return 0.0f;
    }

    float inv_len = 1.0f / (float)len;

    // Instead of recomputing logic, just call tensor_sum with grad_out
    float sum = tensor_sum(input, grad_out, len);  // this fills grad_out with 1s if given

    // If backward, scale the gradients by 1/N
    if (grad_out) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < len; ++i) {
            grad_out[i] *= inv_len;
        }
    }

    return sum * inv_len;
}

void tensor_softmax_ce_backup(
    const float* logits,
    const int* labels,
    const float* grad_loss,
    float* losses,
    float* grad_input,
    float* probs_out,
    size_t batch,
    size_t class_count
) {
    if (!logits || !labels || !losses || !grad_input) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_softmax_ce_intlabel\n");
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
        float* probs_row = probs_out ? probs_out + b * class_count : NULL;
        float* grad_row = grad_input + b * class_count;

        // Find max value
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

        __m256 v_sum_exp = _mm256_set1_ps(sum_exp);  // used in accurate division

        int true_idx = labels[b];
        float prob_true = 0.0f;
        i = 0;
        for (; i + 8 <= class_count; i += 8) {
            __m256 v_probs = _mm256_loadu_ps(probs_row ? probs_row + i : logits_row + i);
            v_probs = _mm256_div_ps(v_probs, v_sum_exp);         // accurate normalization
            v_probs = _mm256_max_ps(v_probs, v_epsilon);         // avoid log(0)

            if (probs_row) _mm256_storeu_ps(probs_row + i, v_probs);

            float tmp[8];
            _mm256_storeu_ps(tmp, v_probs);
            for (int k = 0; k < 8; ++k) {
                size_t idx = i + k;
                float prob = tmp[k];
                if (idx == (size_t)true_idx) prob_true = prob;
                float grad_val = prob - (idx == (size_t)true_idx ? 1.0f : 0.0f);
                if (grad_loss) grad_val *= grad_loss[b];
                grad_row[idx] = grad_val;
            }
        }

        for (; i < class_count; ++i) {
            float prob = probs_row ? probs_row[i] : expf(logits_row[i] - max_val);
            prob /= sum_exp;
            if (prob < epsilon) prob = epsilon;
            if (probs_row) probs_row[i] = prob;

            if ((size_t)i == (size_t)true_idx) prob_true = prob;

            float grad_val = prob - ((size_t)i == (size_t)true_idx ? 1.0f : 0.0f);
            if (grad_loss) grad_val *= grad_loss[b];
            grad_row[i] = grad_val;
        }

        losses[b] = -logf(prob_true < epsilon ? epsilon : prob_true);
        if (grad_loss) losses[b] *= grad_loss[b];
    }
}

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
) {
    if (!logits || !labels || !losses || !grad_input) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_softmax_ce\n");
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
        float* probs_row = probs_out ? probs_out + b * class_count : NULL;
        float* grad_row = grad_input + b * class_count;

        // Step 1: Find max for numerical stability
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

        // Step 2: Compute exp(logits - max) and sum
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

        int true_idx = labels[b];
        float prob_true = 0.0f;

        float on_target = 1.0f;
        float off_target = 0.0f;
        if (use_label_smoothing) {
            on_target = 1.0f - label_smoothing;
            off_target = label_smoothing / (float)(class_count - 1);
        }

        // Step 3: Normalize probs, apply log, compute gradient
        i = 0;
        for (; i + 8 <= class_count; i += 8) {
            __m256 v_probs = _mm256_loadu_ps(probs_row ? probs_row + i : logits_row + i);
            v_probs = _mm256_div_ps(v_probs, v_sum_exp);
            v_probs = _mm256_max_ps(v_probs, v_epsilon); // Avoid log(0)

            if (probs_row) _mm256_storeu_ps(probs_row + i, v_probs);

            float tmp[8];
            _mm256_storeu_ps(tmp, v_probs);
            for (int k = 0; k < 8; ++k) {
                size_t idx = i + k;
                float prob = tmp[k];
                if (idx == (size_t)true_idx) prob_true = prob;

                float target = (idx == (size_t)true_idx) ? on_target : off_target;
                float grad_val = prob - target;
                if (grad_loss) grad_val *= grad_loss[b];
                grad_row[idx] = grad_val;
            }
        }

        for (; i < class_count; ++i) {
            float prob = probs_row ? probs_row[i] : expf(logits_row[i] - max_val);
            prob /= sum_exp;
            if (prob < epsilon) prob = epsilon;
            if (probs_row) probs_row[i] = prob;
            if (i == true_idx) prob_true = prob;

            float target = (i == true_idx) ? on_target : off_target;
            float grad_val = prob - target;
            if (grad_loss) grad_val *= grad_loss[b];
            grad_row[i] = grad_val;
        }

        // Step 4: Compute loss
        float loss = 0.0f;
        for (size_t c = 0; c < class_count; ++c) {
            float prob = probs_row ? probs_row[c] : expf(logits_row[c] - max_val) / sum_exp;
            prob = fmaxf(prob, epsilon);
            float target = (c == true_idx) ? on_target : off_target;
            loss += -target * logf(prob);
        }

        losses[b] = grad_loss ? (loss * grad_loss[b]) : loss;
    }
}