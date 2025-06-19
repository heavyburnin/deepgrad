#ifndef TENSOR_SOFTMAX_CE_H
#define TENSOR_SOFTMAX_CE_H

#include <stddef.h>  // for size_t
#include <stdbool.h> // for bool

// You can define this in the header or externally as a compile-time constant
#define MAX_CLASSES 1024

// Softmax + cross-entropy + gradient (fused)
// - logits: [batch x class_count]
// - labels: one-hot targets, same shape
// - grad_loss: scalar gradient per batch item (or NULL)
// - losses: output loss per batch item
// - grad_input: gradient w.r.t. logits
// - probs_out: optional output for softmax probs (or NULL)
void tensor_softmax_ce(
    const float* logits,
    const float* labels,
    const float* grad_loss,  // Optional: NULL if not provided
    float* losses,
    float* grad_input,
    float* probs_out,        // Optional: NULL if not needed
    size_t batch,
    size_t class_count
);

#endif // TENSOR_SOFTMAX_CE_H
