# DeepGrad

A lightweight, low-level tensor library for building and training neural networks in Python, with optional C‑SIMD acceleration.

---

## 🔧 Features

- **Core Tensor abstraction**: supports autograd, basic ops (`add`, `sub`, `mul`, `div`, `matmul`, `relu`, `sum`, `mean`)
- **Gradient tracking & backprop**: `.backward()`, `.detach()`, `.requires_grad_()`
- **Broadcasting**: full support for NumPy-style broadcasting in Python
- **Optimizers**: built-in SGD
- **Device support**: CPU backend (via C/SIMD) with scaffolding for future GPU extensions
- **Examples**: includes an MLP training script on MNIST-like data

---

## 🚀 Quick Start

1. **Clone and initialize**:
    ```bash
    git clone https://github.com/heavyburnin/deepgrad.git
    cd deepgrad
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2. **Build the C backend** (if you want SIMD acceleration):
    ```bash
    cd ../simd-backend
    mkdir -p build && cd build
    cmake .. && make
    # builds libsimd_tensor_backend.so
    ```

3. **Run the example training script**:
    ```bash
    cd ../deepgrad  # project root
    python3 -m deepgrad.examples.train
    ```

    ✅ This will:
    - Convert `mnist_train.csv` to `.bin`
    - Initialize an MLP
    - Train for a few epochs, printing loss & accuracy

---

## 🛠️ Project Organization
```bash
deepgrad/
├── tensor.py – Core Tensor class + ops
├── ops.py – Operator registry (forward/backward function names)
├── backend.py – ctypes bindings to libsimd_tensor_backend.so
├── utils.py – Pure-Python helper functions
└── examples/
├── model.py – Model definitions (e.g. MLP)
└── train.py – Example training script
```

🧩 The C backend is assumed to be built at:

- `../simd-backend/build/libsimd_tensor_backend.so`

---

## 📦 Install for Development
To avoid import and path issues later, install DeepGrad as an editable package:

1. Create a `setup.py` or add a `pyproject.toml`
2. Install in dev mode:
    ```bash
    pip install -e .
    ```

---

## 🧠 Features & Usage

- **Basic usage**:
    ```python
    from deepgrad.tensor import Tensor
    import array

    a = Tensor(array.array('f', [1,2,3]), requires_grad=True, shape=(3,))
    b = Tensor(array.array('f', [0.1]), requires_grad=False, shape=(1,))
    c = a + b
    loss = c.sum()
    loss.backward()
    ```

- **Optimizer**:
    ```python
    from deepgrad.model import MLP
    from deepgrad.optimizer import SGD

    model = MLP(...)
    optimizer = SGD(model.parameters(), lr=0.01)
    optimizer.step()
    ```

---

## 📊 Additions & TODOs

- ⚡ **GPU support**: Extend `Tensor` to use `device="cuda"` and stub hooks in `backend.py`
- 🧪 **More ops**: Support `log`, `exp`, `matmul-backward`, etc.
- ✅ **Unit tests**: Validate correctness and gradient checking
- 📚 **Data loaders**: Built-in MNIST/CIFAR support
- 🎨 **Improved optimizers**: Momentum, Adam, RMSprop

---

## 📝 Contributing

Feel free to:

- Add new operators (`register_op` in `ops.py`)
- Write unit tests (`tests/` directory)
- Extend for GPU/backends
- Open issues or PRs here on GitHub

---

## 🔐 License
This project is licensed under the [MIT License](LICENSE).

---

## ❤️ Credits

Built by **heavyburnin**, with inspiration from tinygrad and PyTorch tensor broadcasting & autograd systems.
