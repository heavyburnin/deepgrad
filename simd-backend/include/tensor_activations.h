#ifndef TENSOR_ACTIVATIONS_H
#define TENSOR_ACTIVATIONS_H

#include <stddef.h>

#define VEC_SIZE 8  // AVX2 = 8 floats per vector

// ReLU activation
void tensor_relu(const float* a, float* out, size_t n);

// ReLU gradient (in-place add to grad_input)
void tensor_relu_backward(const float* grad_output, const float* input, float* grad_input, size_t n);

#endif // TENSOR_ACTIVATIONS_H