// tensor_reductions.h

#ifndef TENSOR_REDUCTIONS_H
#define TENSOR_REDUCTIONS_H

#include <stddef.h>  // for size_t

// Compute sum of all elements in a float array
float tensor_sum(const float* input, size_t len);

// Compute mean of all elements in a float array
float tensor_mean(const float* input, size_t len);

#endif // TENSOR_REDUCTIONS_H