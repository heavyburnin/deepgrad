// tensor_math.c

#include "tensor_math.h"   // Your function declarations
#include "tensor_utils.h"  // For AVX helpers (if used later)
#include <immintrin.h>     // For __m256, _mm256_* intrinsics
#include <stddef.h>        // For size_t
#include <stdio.h>         // For fprintf, stderr
#include <omp.h>           // For OpenMP pragmas

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