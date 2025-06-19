#ifndef TENSOR_BROADCAST_H
#define TENSOR_BROADCAST_H

#include <stddef.h>

// Broadcast a single row (1, N) across B rows → (B, N)
void tensor_broadcast_row(const float* input, float* output, size_t B, size_t N);

// Broadcast a single column (B, 1) across N columns → (B, N)
void tensor_broadcast_col(const float* input, float* output, size_t B, size_t N);

// Reduce gradients over broadcasted axes (inverse of broadcast)
void tensor_unbroadcast_sum_axes(
    const float* grad,          // input gradient (from output)
    float* out,                 // reduced gradient matching original tensor
    const size_t* shape_out,    // shape of the unbroadcasted tensor
    const size_t* strides_grad, // strides for the broadcasted gradient
    const size_t* strides_out,  // strides for the output tensor
    size_t ndim,                // number of dimensions
    size_t total_grad,          // total size of grad (B*N...)
    size_t total_out            // total size of out
);

#endif // TENSOR_BROADCAST_H