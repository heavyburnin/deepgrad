# SIMD Tensor Backend

A high-performance tensor operations library for deep learning, optimized with SIMD instructions (AVX2/FMA) and OpenMP parallelization.

## Features

- Vectorized element-wise operations (add, subtract, multiply, divide)
- Optimized activation functions (ReLU, exponential)
- Fast matrix multiplication with cache-friendly memory access patterns
- Efficient softmax cross-entropy implementation
- Reduction operations (sum, mean)
- JIT batching with automatic parallelization
- AVX2 and FMA instruction set optimizations

## Requirements

- C11 compatible compiler
- CMake 3.10+
- CPU with AVX2 and FMA support
- OpenMP (optional, for parallelization)

## Building

```bash
mkdir -p build
cd build
cmake ..
make
```

The library will be built in the `build` directory.

## Installation

```bash
sudo make install
```

This will install the library and headers to your system.

## Usage

Include the header in your C/C++ code:

```c
#include <tensor_ops.h>
```

Link against the library:

```bash
gcc -o your_program your_program.c -lsimd_tensor_backend -lm
```

### Example

```c
#include <tensor_ops.h>
#include <stdio.h>

int main() {
    // Create input tensors
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float result[4];
    
    // Perform element-wise addition
    tensor_add(a, b, result, 4);
    
    // Print results
    for (int i = 0; i < 4; i++) {
        printf("%.1f + %.1f = %.1f\n", a[i], b[i], result[i]);
    }
    
    return 0;
}
```

## API Reference

### Element-wise Operations

```c
void tensor_add(const float* a, const float* b, float* out, size_t n);
void tensor_sub(const float* a, const float* b, float* out, size_t n);
void tensor_mul(const float* a, const float* b, float* out, size_t n);
void tensor_div(const float* a, const float* b, float* out, size_t n);
```

### Activation Functions

```c
void tensor_exp(const float* a, float* out, size_t n);
void tensor_relu(const float* a, float* out, size_t n);
```

### Matrix Operations

```c
void tensor_matmul(const float* A, const float* B, float* C, 
                   size_t M, size_t K, size_t N);
```

### Loss Functions

```c
void tensor_softmax_cross_entropy_with_probs(const float* logits,
                                             const float* labels,
                                             float* loss_out,
                                             float* probs_out,
                                             size_t class_count);
```

### Reduction Operations

```c
float tensor_sum(const float* input, size_t len);
float tensor_mean(const float* input, size_t len);
```

### Batched Operations

```c
void tensor_op_jit_2in(void (*op)(const float*, const float*, float*, size_t),
                       const float* a, const float* b, float* out,
                       size_t batch, size_t stride);

void tensor_op_jit_1in(void (*op)(const float*, float*, size_t),
                       const float* a, float* out,
                       size_t batch, size_t stride);

void tensor_matmul_batch_jit(const float* A, const float* B, float* C,
                             size_t batch, size_t M, size_t K, size_t N);
```

## Performance

This library is optimized for modern x86-64 processors with AVX2 and FMA instruction sets. Key optimizations include:

- SIMD vectorization using AVX2 instructions
- Cache-friendly memory access patterns
- Loop unrolling for better instruction-level parallelism
- Multi-threading with OpenMP for batch operations
- Custom fast approximations for transcendental functions

## License

[MIT License](LICENSE)
