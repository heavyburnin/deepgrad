# DeepGrad

A lightweight, low-level tensor library for building and training neural networks in Python, with C-SIMD acceleration for high performance.

---

## 🔧 Features

- **Core Tensor Abstraction**: Supports autograd, basic arithmetic (`add`, `sub`, `mul`, `div`, `pow`), matrix multiplication (`matmul`), and element-wise operations (`relu`, `tanh`, `log_softmax`).
- **Convolutional Operations**: Includes 2D convolution (`conv2d`), max pooling (`maxpool2d`), and average pooling (`avgpool2d`) for CNNs.
- **Loss Functions**: Cross-entropy loss with optional label smoothing (`cross_entropy`).
- **Dropout**: Supports dropout for regularization during training.
- **Gradient Tracking & Backpropagation**: Methods like `.backward()`, `.detach()`, `.requires_grad_()`, and `.release()` for efficient autograd.
- **Broadcasting**: Full support for NumPy-style broadcasting in Python.
- **Optimizers**: Built-in SGD with plans for Momentum, Adam, and RMSprop.
- **Device Support**: CPU backend with SIMD acceleration (via C backend).
- **Utilities**: Tensor creation functions (`zeros`, `ones`, `rand`, `randn`) and reshaping (`reshape`, `flatten`, `permute`).
- **Examples**: Includes an MNISTNet training script on the MNIST dataset.

---

## 🚀 Quick Start

1. **Clone and Initialize**:
    ```bash
    git clone https://github.com/heavyburnin/deepgrad.git
    cd deepgrad
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2. **Build the C Backend**:
    ```bash
    cd ../simd-backend
    mkdir -p build && cd build
    cmake .. && make
    # Builds libsimd_tensor_backend.so
    ```

3. **Run the Example Training Script**:
    ```bash
    cd ../deepgrad  # Project root
    python3 -m deepgrad.examples.train
    ```

    ✅ This will:
    - Convert `mnist_train.csv` to `.bin`
    - Initialize an ConvNet
    - Train for a few epochs, printing loss and accuracy

---

## 🛠️ Project Organization

```bash
deepgrad/
├── tensor.py       – Core Tensor class with autograd and operations
├── ops.py         – Operator registry (forward/backward function names)
├── backend.py     – ctypes bindings to libsimd_tensor_backend.so
├── broadcast.py   – Broadcasting utilities
├── utils.py       – Pure-Python helper functions
├── optimizer.py   – Optimizer implementations (e.g., SGD)
├── model.py       – Model definitions (e.g., MNISTNet, FashionNet)
└── examples/
    ├── model.py   – Example model definitions
    └── train.py   – Example training script
```

🧩 The C backend is assumed to be built at:

- `../simd-backend/build/libsimd_tensor_backend.so`

---


## 🧠 Features & Usage

### Basic Tensor Operations
```python
from deepgrad.tensor import Tensor

# Create tensors
a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
b = Tensor([0.1], requires_grad=False)

# Perform operations
c = a + b  # Broadcasting
loss = c.sum()
loss.backward()
print(a.grad)  # Gradient of loss w.r.t. a
```

### Convolutional Neural Networks
```python
# 2D Convolution
input = Tensor.randn((1, 3, 28, 28), requires_grad=True)  # Batch, Channels, Height, Width
weight = Tensor.randn((16, 3, 3, 3))  # Out_channels, In_channels, Kernel_h, Kernel_w
bias = Tensor.zeros((16,))
output = input.conv2d(weight, bias, stride=(1, 1), padding=(1, 1))
```

### Pooling and Activation
```python
pooled = output.maxpool2d(kernel_size=(2, 2), stride=(2, 2))
activated = pooled.relu()
```

### Loss and Optimization
```python
from deepgrad.model import MNISTNet
from deepgrad.optimizer import SGD

model = MNISTNet(input_dim=784, hidden_dims=[128, 64], output_dim=10)
optimizer = SGD(model.parameters(), lr=0.01)

# Forward pass
pred = model(input)
target = Tensor([0, 1, 2, ...], shape=(batch_size,))
loss = pred.cross_entropy(target, label_smoothing=0.1)

# Backward pass
loss.backward()
optimizer.step()
```

### Dropout for Regularization
```python
x = Tensor.randn((10, 100), requires_grad=True)
x_dropped = x.dropout(p=0.5)  # Drops 50% of elements during training
```

---

## 📊 Additions & TODOs

- ⚡ **GPU Support**: Extend `Tensor` to support `device="cuda"` with hooks in `backend.py`.
- 🧪 **More Operations**: Add `log`, `exp`, `sigmoid`, and advanced matrix operations.
- ✅ **Unit Tests**: Implement gradient checking and correctness tests in `tests/`.
- 📚 **Data Loaders**: Add built-in support for MNIST, CIFAR-10, and other datasets.
- 🎨 **Improved Optimizers**: Implement Momentum, Adam, and RMSprop.
- 🔄 **Graph Optimization**: Enhance `release()` and `release_graph()` for memory efficiency.

---

## 📝 Contributing

Contributions are welcome! You can:

- Add new operators in `ops.py` using `register_op`.
- Write unit tests in the `tests/` directory.
- Extend backend support (e.g., GPU or other accelerators).
- Open issues or PRs on GitHub.

---

## 🔐 License

This project is licensed under the [MIT License](LICENSE).

---

## ❤️ Credits

Built by **heavyburnin**, with inspiration from tinygrad and PyTorch tensor broadcasting and autograd systems.