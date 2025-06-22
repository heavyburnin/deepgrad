// tensor_math.c

#include "tensor_math.h"   // Your function declarations
#include "tensor_utils.h"  // For AVX helpers (if used later)
#include <immintrin.h>     // For __m256, _mm256_* intrinsics
#include <stddef.h>        // For size_t
#include <stdio.h>         // For fprintf, stderr
#include <omp.h>           // For OpenMP pragmas
#include <math.h>

static inline __m256 exp256_ps(__m256 x) {
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);  // log2(e)
    const __m256 c0 = _mm256_set1_ps(0.99992522f);
    const __m256 c1 = _mm256_set1_ps(0.69583354f);
    const __m256 c2 = _mm256_set1_ps(0.22606716f);
    const __m256 c3 = _mm256_set1_ps(0.078024523f);
    const __m256 c4 = _mm256_set1_ps(0.014330218f);
    const __m256 c5 = _mm256_set1_ps(0.0026592043f);

    __m256 fx = _mm256_mul_ps(x, log2e);
    fx = _mm256_max_ps(_mm256_set1_ps(-126.0f), fx);  // Clamp underflow
    fx = _mm256_min_ps(_mm256_set1_ps(129.0f), fx);   // Clamp overflow

    __m256i emm0 = _mm256_cvttps_epi32(fx);
    __m256 tmp = _mm256_cvtepi32_ps(emm0);

    __m256 r = _mm256_sub_ps(fx, tmp);

    // Estrin's method
    __m256 y = c5;
    y = _mm256_fmadd_ps(y, r, c4);
    y = _mm256_fmadd_ps(y, r, c3);
    y = _mm256_fmadd_ps(y, r, c2);
    y = _mm256_fmadd_ps(y, r, c1);
    y = _mm256_fmadd_ps(y, r, c0);

    __m256 pow2n = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_add_epi32(emm0, _mm256_set1_epi32(127)), 23)
    );

    return _mm256_mul_ps(pow2n, y);
}

// Approximate log(x) for x > 0 using AVX2
// Based on: log(x) = log(m * 2^e) = log(m) + e * log(2)
// m in [0.5, 1.0) for normalized float
static inline __m256 log256_ps(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);

    // Mask out exponent and mantissa
    __m256i ix = _mm256_castps_si256(x);
    __m256i exp = _mm256_srli_epi32(ix, 23);
    exp = _mm256_sub_epi32(exp, _mm256_set1_epi32(127));

    // Normalize mantissa to [0.5, 1)
    __m256i mant_mask = _mm256_set1_epi32(0x007FFFFF);
    __m256i norm = _mm256_or_si256(_mm256_and_si256(ix, mant_mask), _mm256_set1_epi32(0x3f000000));
    __m256 m = _mm256_castsi256_ps(norm);

    // Polynomial approximation of log(m)
    // log(m) ≈ c1*(m-1) + c2*(m-1)^2 + c3*(m-1)^3 + ...
    __m256 r = _mm256_sub_ps(m, one);
    // __m256 r2 = _mm256_mul_ps(r, r);
    // __m256 r3 = _mm256_mul_ps(r2, r);

    const __m256 c1 = _mm256_set1_ps(0.9999964239f);
    const __m256 c2 = _mm256_set1_ps(-0.4998741238f);
    const __m256 c3 = _mm256_set1_ps(0.3317990258f);
    const __m256 c4 = _mm256_set1_ps(-0.2407338084f);
    const __m256 c5 = _mm256_set1_ps(0.1676540711f);

    __m256 y = c5;
    y = _mm256_fmadd_ps(y, r, c4);
    y = _mm256_fmadd_ps(y, r, c3);
    y = _mm256_fmadd_ps(y, r, c2);
    y = _mm256_fmadd_ps(y, r, c1);
    y = _mm256_mul_ps(y, r);

    // e * ln(2)
    __m256 e = _mm256_cvtepi32_ps(exp);
    const __m256 ln2 = _mm256_set1_ps(0.69314718056f);
    y = _mm256_fmadd_ps(e, ln2, y);
    return y;
}

void tensor_add(const float* a, const float* b, float* out, size_t n, size_t batch_size) {
    if (!a || !b || !out) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_add_batch\n");
        return;
    }

    // Total number of elements across all batches
    const size_t total_elements = batch_size * n;
    const size_t vec_end = total_elements - (total_elements % 8);

    // Parallelize over batches and elements
    #pragma omp parallel for schedule(static) if (total_elements > 10000)
    for (size_t i = 0; i < vec_end; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
    }

    // Handle remaining elements (if total_elements is not divisible by 8)
    for (size_t i = vec_end; i < total_elements; i++) {
        out[i] = a[i] + b[i];
    }
}

void tensor_add_grad(const float* dout, const float* a, const float* b, float* da, float* db, size_t n, size_t batch_size) {
    if (!dout || !da || !db) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_add_grad_batch\n");
        return;
    }

    // Total number of elements across all batches
    const size_t total_elements = batch_size * n;
    const size_t vec_end = total_elements - (total_elements % 8);

    #pragma omp parallel for schedule(static) if (total_elements > 10000)
    for (size_t i = 0; i < vec_end; i += 8) {
        __m256 v_dout = _mm256_loadu_ps(dout + i);

        __m256 v_da = _mm256_loadu_ps(da + i);
        __m256 v_db = _mm256_loadu_ps(db + i);

        _mm256_storeu_ps(da + i, _mm256_add_ps(v_da, v_dout));
        _mm256_storeu_ps(db + i, _mm256_add_ps(v_db, v_dout));
    }

    // Handle remaining elements
    for (size_t i = vec_end; i < total_elements; i++) {
        da[i] += dout[i];
        db[i] += dout[i];
    }
}

void tensor_sub(const float* a, const float* b, float* out, size_t n, size_t batch_size) {
    if (!a || !b || !out) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_sub_batch\n");
        return;
    }

    size_t total_elements = batch_size * n;
    size_t vec_end = total_elements - (total_elements % 8);

    #pragma omp parallel for schedule(static) if (total_elements > 10000)
    for (size_t i = 0; i < vec_end; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_sub_ps(va, vb));
    }

    for (size_t i = vec_end; i < total_elements; i++) {
        out[i] = a[i] - b[i];
    }
}

void tensor_sub_grad(const float* dout, const float* a, const float* b, float* da, float* db, size_t n, size_t batch_size) {
    if (!dout || !da || !db) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_sub_grad_batch\n");
        return;
    }

    size_t total_elements = batch_size * n;
    size_t vec_end = total_elements - (total_elements % 8);

    #pragma omp parallel for schedule(static) if (total_elements > 10000)
    for (size_t i = 0; i < vec_end; i += 8) {
        __m256 v_dout = _mm256_loadu_ps(dout + i);

        __m256 v_da = _mm256_loadu_ps(da + i);
        __m256 v_db = _mm256_loadu_ps(db + i);

        _mm256_storeu_ps(da + i, _mm256_add_ps(v_da, v_dout));
        _mm256_storeu_ps(db + i, _mm256_sub_ps(v_db, v_dout));
    }

    for (size_t i = vec_end; i < total_elements; i++) {
        da[i] += dout[i];
        db[i] -= dout[i];
    }
}

void tensor_mul(const float* a, const float* b, float* out, size_t n, size_t batch_size) {
    if (!a || !b || !out) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_mul_batch\n");
        return;
    }

    size_t total_elements = batch_size * n;
    size_t vec_end = total_elements - (total_elements % 8);

    #pragma omp parallel for schedule(static) if (total_elements > 10000)
    for (size_t i = 0; i < vec_end; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
    }

    for (size_t i = vec_end; i < total_elements; i++) {
        out[i] = a[i] * b[i];
    }
}

void tensor_mul_grad(const float* dout, const float* a, const float* b, float* da, float* db, size_t n, size_t batch_size) {
    if (!dout || !da || !db) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_mul_grad_batch\n");
        return;
    }

    size_t total_elements = batch_size * n;
    size_t vec_end = total_elements - (total_elements % 8);

    #pragma omp parallel for schedule(static) if (total_elements > 10000)
    for (size_t i = 0; i < vec_end; i += 8) {
        __m256 v_dout = _mm256_loadu_ps(dout + i);
        __m256 v_a = _mm256_loadu_ps(a + i);
        __m256 v_b = _mm256_loadu_ps(b + i);

        __m256 v_da = _mm256_mul_ps(v_dout, v_b);
        __m256 v_db = _mm256_mul_ps(v_dout, v_a);

        _mm256_storeu_ps(da + i, _mm256_add_ps(_mm256_loadu_ps(da + i), v_da));
        _mm256_storeu_ps(db + i, _mm256_add_ps(_mm256_loadu_ps(db + i), v_db));
    }

    for (size_t i = vec_end; i < total_elements; i++) {
        da[i] += dout[i] * b[i];
        db[i] += dout[i] * a[i];
    }
}

void tensor_div(const float* a, const float* b, float* out, size_t n, size_t batch_size) {
    if (!a || !b || !out) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_div_batch\n");
        return;
    }

    size_t total_elements = batch_size * n;
    size_t vec_end = total_elements - (total_elements % 8);

    #pragma omp parallel for schedule(static) if (total_elements > 10000)
    for (size_t i = 0; i < vec_end; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_div_ps(va, vb));
    }

    for (size_t i = vec_end; i < total_elements; i++) {
        out[i] = a[i] / b[i];
    }
}

void tensor_div_grad(const float* dout, const float* a, const float* b, float* da, float* db, size_t n, size_t batch_size) {
    if (!dout || !da || !db) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_div_grad_batch\n");
        return;
    }

    size_t total_elements = batch_size * n;
    size_t vec_end = total_elements - (total_elements % 8);

    #pragma omp parallel for schedule(static) if (total_elements > 10000)
    for (size_t i = 0; i < vec_end; i += 8) {
        __m256 v_dout = _mm256_loadu_ps(dout + i);
        __m256 v_a = _mm256_loadu_ps(a + i);
        __m256 v_b = _mm256_loadu_ps(b + i);

        __m256 v_da = _mm256_div_ps(v_dout, v_b);

        __m256 v_db = _mm256_mul_ps(v_dout, v_a);
        v_db = _mm256_div_ps(v_db, _mm256_mul_ps(v_b, v_b));
        v_db = _mm256_sub_ps(_mm256_setzero_ps(), v_db);

        _mm256_storeu_ps(da + i, _mm256_add_ps(_mm256_loadu_ps(da + i), v_da));
        _mm256_storeu_ps(db + i, _mm256_add_ps(_mm256_loadu_ps(db + i), v_db));
    }

    for (size_t i = vec_end; i < total_elements; i++) {
        da[i] += dout[i] / b[i];
        db[i] -= dout[i] * a[i] / (b[i] * b[i]);
    }
}

void tensor_exp(const float* a, float* out, size_t n, size_t batch_size) {
    if (!a || !out) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_exp\n");
        return;
    }

    size_t total = batch_size * n;
    size_t vec_end = total - (total % 8);

    #pragma omp parallel for schedule(static) if (total > 10000)
    for (size_t i = 0; i < vec_end; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(out + i, exp256_ps(va));
    }

    for (size_t i = vec_end; i < total; ++i) {
        out[i] = expf(a[i]);
    }
}

void tensor_exp_grad(const float* dout, const float* a, float* da, size_t n, size_t batch_size) {
    if (!dout || !da) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_exp_grad\n");
        return;
    }

    size_t total = batch_size * n;
    size_t vec_end = total - (total % 8);

    #pragma omp parallel for schedule(static) if (total > 10000)
    for (size_t i = 0; i < vec_end; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vout = exp256_ps(va);
        __m256 vdout = _mm256_loadu_ps(dout + i);
        __m256 vgrad = _mm256_mul_ps(vdout, vout);
        _mm256_storeu_ps(da + i, _mm256_add_ps(_mm256_loadu_ps(da + i), vgrad));
    }

    for (size_t i = vec_end; i < total; ++i) {
        da[i] += dout[i] * expf(a[i]);
    }
}

void tensor_pow(const float* base, const float* exponent, float* out, size_t n, size_t batch_size) {
    if (!base || !exponent || !out) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_pow\n");
        return;
    }

    size_t total = batch_size * n;
    size_t vec_end = total - (total % 8);

    #pragma omp parallel for schedule(static) if (total > 10000)
    for (size_t i = 0; i < vec_end; i += 8) {
        __m256 vb = _mm256_loadu_ps(base + i);
        __m256 ve = _mm256_loadu_ps(exponent + i);

        // Avoid log(0) by max(base, ε)
        __m256 vb_clamped = _mm256_max_ps(vb, _mm256_set1_ps(1e-10f));
        __m256 vlog = log256_ps(vb_clamped);
        __m256 vexp = _mm256_mul_ps(ve, vlog);
        __m256 vpow = exp256_ps(vexp);
        _mm256_storeu_ps(out + i, vpow);
    }

    for (size_t i = vec_end; i < total; ++i) {
        float b = fmaxf(base[i], 1e-10f);
        out[i] = powf(b, exponent[i]);
    }
}

void tensor_pow_grad(const float* dout, const float* base, const float* exponent,
                     float* dbase, float* dexp, size_t n, size_t batch_size) {
    if (!dout || !dbase || !dexp) {
        fprintf(stderr, "Error: NULL pointer passed to tensor_pow_grad\n");
        return;
    }

    size_t total = batch_size * n;
    size_t vec_end = total - (total % 8);

    #pragma omp parallel for schedule(static) if (total > 10000)
    for (size_t i = 0; i < vec_end; i += 8) {
        __m256 vb = _mm256_loadu_ps(base + i);
        __m256 ve = _mm256_loadu_ps(exponent + i);
        __m256 vdout = _mm256_loadu_ps(dout + i);

        __m256 vb_clamped = _mm256_max_ps(vb, _mm256_set1_ps(1e-10f));
        __m256 vlogb = log256_ps(vb_clamped);
        __m256 pow_val = exp256_ps(_mm256_mul_ps(ve, vlogb));

        // dbase = dout * e * base^(e-1) = dout * e * pow(base, e-1)
        __m256 e_minus_1 = _mm256_sub_ps(ve, _mm256_set1_ps(1.0f));
        __m256 base_pow_e_minus_1 = exp256_ps(_mm256_mul_ps(e_minus_1, vlogb));
        __m256 dbase_vec = _mm256_mul_ps(vdout, _mm256_mul_ps(ve, base_pow_e_minus_1));

        // dexp = dout * pow(base, e) * log(base)
        __m256 dexp_vec = _mm256_mul_ps(vdout, _mm256_mul_ps(pow_val, vlogb));

        _mm256_storeu_ps(dbase + i, _mm256_add_ps(_mm256_loadu_ps(dbase + i), dbase_vec));
        _mm256_storeu_ps(dexp + i, _mm256_add_ps(_mm256_loadu_ps(dexp + i), dexp_vec));
    }

    for (size_t i = vec_end; i < total; ++i) {
        float b = fmaxf(base[i], 1e-10f);
        float e = exponent[i];
        float pow_val = powf(b, e);
        dbase[i] += dout[i] * e * powf(b, e - 1.0f);
        dexp[i] += dout[i] * pow_val * logf(b);
    }
}