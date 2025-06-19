// tensor_avx_init.h

#ifndef TENSOR_AVX_INIT_H
#define TENSOR_AVX_INIT_H

#include <stddef.h>
#include <stdbool.h>

// Check AVX2 hardware support
int has_avx2(void);

// Initialize the SIMD tensor backend (threads, AVX checks)
int tensor_ops_init(void);

#endif // TENSOR_AVX_INIT_H