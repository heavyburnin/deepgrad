"""
DeepGrad Tensor Library
=======================

A high-performance tensor library with autograd support, accelerated by a SIMD backend.
Provides NumPy-like functionality for tensor operations with automatic differentiation.

Version: 1.0.0
License: MIT
Author: Your Name (or Organization)

Key Features:
- Basic arithmetic (add, sub, mul, div, pow)
- Matrix multiplication
- Broadcasting and shape inference
- Convolution, pooling, and activation functions (e.g., ReLU)
- Loss functions like cross-entropy
- Backpropagation via `.backward()`

Usage Example:
```python
from deepgrad.tensor import Tensor

# Create tensors
a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
b = Tensor([4.0, 5.0, 6.0])

# Perform operations
c = a * b
loss = c.sum()
loss.backward()

print(a.grad)  # Gradient of loss w.r.t `a`
```

Backend:
- Uses SIMD-accelerated C functions via `SimdTensorBackend` for performance.
"""

from typing import Union, Tuple, List, Set, Optional
from deepgrad.backend import SimdTensorBackend
from ctypes import c_float, c_size_t, Array, cast, POINTER, c_int
from deepgrad.broadcast import compute_broadcast_shape, broadcast_to_shape, unbroadcast_grad
from deepgrad.ops import get_op_names
from functools import reduce
from operator import mul
import random
import math

__version__ = "1.0.0"
__all__ = ["Tensor", "zeros", "ones", "rand", "randn"]

class Tensor:
    """
    A multi-dimensional array with autograd support for automatic differentiation.

    Args:
        data: Input data as a list, tuple, or ctypes c_float array.
        requires_grad (bool): If True, tracks gradients for backpropagation.
        grad: Initial gradient buffer (optional, ctypes c_float array).
        shape: Shape of the tensor (optional, inferred if None).
        size: Total number of elements (optional, inferred if None).

    Attributes:
        data: ctypes c_float array containing the tensor's data.
        shape: Tuple representing the tensor's shape.
        size: Total number of elements.
        requires_grad: Whether gradients are tracked.
        grad: Gradient buffer (ctypes c_float array) or None.
        ndim: Number of dimensions.

    Raises:
        ValueError: If data type or shape is invalid.
    """
    def __init__(self, data, requires_grad: bool = False, grad=None, shape: Optional[Tuple[int, ...]] = None, size: Optional[int] = None):
        if isinstance(data, (list, tuple)):
            self.data = (c_float * len(data))(*data)
            self.owns_data = True
        elif isinstance(data, Array) and data._type_ in (c_float, c_int):
            self.data = data
            self.owns_data = False
        else:
            raise ValueError(f"data must be a list, tuple, or c_float/c_int array, got {type(data)}")

        self.requires_grad = requires_grad
        self._grad = grad
        self._backward = None
        self._prev = []

        if shape is not None:
            self.shape = shape
        else:
            if size is not None:
                self.shape = (size,)
            else:
                self.shape = (len(data),)

        expected_size = 1
        for dim in self.shape:
            if dim <= 0:
                raise ValueError(f"Shape dimensions must be positive, got {self.shape}")
            expected_size *= dim

        if size is not None:
            self.size = size
        else:
            self.size = len(data)

        if self.size != expected_size:
            raise ValueError(f"Shape {self.shape} incompatible with data size {self.size}")

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'data' in state and hasattr(state['data'], '__len__') and isinstance(state['data'], Array):
            state['data'] = [state['data'][i] for i in range(len(state['data']))]
        if '_grad' in state and state['_grad'] is not None:
            if hasattr(state['_grad'], '__len__') and isinstance(state['_grad'], Array):
                state['_grad'] = [state['_grad'][i] for i in range(len(state['_grad']))]
        return state

    def __setstate__(self, state):
        if 'data' in state and isinstance(state['data'], list):
            state['data'] = (c_float * len(state['data']))(*state['data'])
        if '_grad' in state and isinstance(state['_grad'], list):
            state['_grad'] = (c_float * len(state['_grad']))(*state['_grad'])
        self.__dict__.update(state)

    def __len__(self):
        return self.shape[0] if self.shape else 1

    @property
    def grad(self):
        """Gradient buffer, initialized with zeros if None and requires_grad is True."""
        if not self.requires_grad:
            return None
        if self._grad is None and hasattr(self, '_backward') and self._backward is not None:
            self._grad = (c_float * self.size)(0.0)
        return self._grad
    
    @grad.setter
    def grad(self, value):
        self._grad = value

    @property
    def ndim(self) -> int:
        """Number of dimensions in the tensor."""
        return len(self.shape)

    def _apply_op(self, other: 'Tensor', op_name: str, grad_fn_name: str) -> 'Tensor':
        if not isinstance(other, Tensor):
            other_data = (c_float * 1)(float(other))
            other = Tensor(other_data, shape=(1,), requires_grad=False)

        s1, s2 = self.shape, other.shape
        max_rank = max(len(s1), len(s2))
        shape1 = (1,) * (max_rank - len(s1)) + s1
        shape2 = (1,) * (max_rank - len(s2)) + s2

        out_shape = compute_broadcast_shape(shape1, shape2)

        a_broadcasted = (
            self.data if self.shape == out_shape 
            else broadcast_to_shape(self.data, self.shape, out_shape, self.size)
        )
        b_broadcasted = (
            other.data if other.shape == out_shape 
            else broadcast_to_shape(other.data, other.shape, out_shape, other.size)
        )

        out_size = reduce(mul, out_shape, 1)
        out_data = (c_float * out_size)()

        if len(out_shape) > 1:
            batch_size = out_shape[0]
            n = reduce(mul, out_shape[1:], 1)
            use_batch = True
        else:
            batch_size = 1
            n = out_size
            use_batch = False

        getattr(SimdTensorBackend, op_name)(
            a_broadcasted,
            b_broadcasted,
            out_data,
            n,
            batch_size if use_batch else 0
        )

        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad, shape=out_shape)

        if out.requires_grad:
            out._cached_a_broadcasted = a_broadcasted
            out._cached_b_broadcasted = b_broadcasted
            out._batch_info = (n, batch_size if use_batch else None)

            def _backward():
                out_grad = out.grad
                a_broadcasted = out._cached_a_broadcasted
                b_broadcasted = out._cached_b_broadcasted
                n, batch_size_cached = out._batch_info
                use_batch_cached = batch_size_cached is not None

                self_grad = (c_float * out_size)()
                other_grad = (c_float * out_size)()

                grad_fn = getattr(SimdTensorBackend, grad_fn_name)

                if use_batch_cached:
                    grad_fn(out_grad, a_broadcasted, b_broadcasted, self_grad, other_grad, n, batch_size_cached)
                else:
                    grad_fn(out_grad, a_broadcasted, b_broadcasted, self_grad, other_grad, n, 0)

                if self.requires_grad:
                    grad_contrib = unbroadcast_grad(self_grad, out.shape, self.shape)
                    if len(grad_contrib) != self.size:
                        raise ValueError(f"Gradient size mismatch: expected {self.size}, got {len(grad_contrib)}")
                    if self._grad is None:
                        self._grad = grad_contrib
                    else:
                        SimdTensorBackend.tensor_add_inplace(self._grad, grad_contrib, self.size)

                if other.requires_grad:
                    grad_contrib = unbroadcast_grad(other_grad, out.shape, other.shape)
                    if len(grad_contrib) != other.size:
                        raise ValueError(f"Gradient size mismatch: expected {other.size}, got {len(grad_contrib)}")
                    if other._grad is None:
                        other._grad = grad_contrib
                    else:
                        SimdTensorBackend.tensor_add_inplace(other._grad, grad_contrib, other.size)

            out._backward = _backward
            out._prev = [self, other]

        return out

    def _binary_op(self, other: Union['Tensor', float, int], op_name: str) -> 'Tensor':
        forward_fn, backward_fn = get_op_names(op_name)
        return self._apply_op(other, forward_fn, backward_fn)

    def __add__(self, other): return self._binary_op(other, 'add')
    def __radd__(self, other): return self.__add__(other)
    def __sub__(self, other): return self._binary_op(other, 'sub')
    def __rsub__(self, other):
        other_t = other if isinstance(other, Tensor) else Tensor([other], shape=(1,), requires_grad=False)
        return other_t.__sub__(self)
    def __mul__(self, other): return self._binary_op(other, 'mul')
    def __rmul__(self, other): return self.__mul__(other)
    def __truediv__(self, other): return self._binary_op(other, 'div')
    def __rtruediv__(self, other):
        other_t = other if isinstance(other, Tensor) else Tensor([other], shape=(1,), requires_grad=False)
        return other_t.__truediv__(self)
    def __pow__(self, other): return self._binary_op(other, 'pow')
    def __rpow__(self, other):
        other_t = other if isinstance(other, Tensor) else Tensor([other], shape=(1,), requires_grad=False)
        return other_t.__pow__(self)

    def reshape(self, new_shape: Tuple[int, ...]) -> 'Tensor':
        """
        Reshapes the tensor to the specified shape, preserving the total number of elements.

        Args:
            new_shape: Desired shape, may include one -1 for inferred dimension.

        Returns:
            A new Tensor with the specified shape.

        Raises:
            ValueError: Áf the new shape is incompatible with the tensor's size.
        """
        inferred_index = -1
        known_product = 1
        for i, dim in enumerate(new_shape):
            if dim == -1:
                if inferred_index != -1:
                    raise ValueError("Only one dimension can be inferred")
                inferred_index = i
            else:
                known_product *= dim

        total = reduce(mul, self.shape, 1)
        if inferred_index != -1:
            if total % known_product != 0:
                raise ValueError("Invalid shape for inference")
            new_shape = list(new_shape)
            new_shape[inferred_index] = total // known_product

        new_shape = tuple(new_shape)
        if reduce(mul, new_shape, 1) != total:
            raise ValueError(f"Reshape must preserve size (got {new_shape}, expected {total})")

        out = Tensor(
            data=self.data,
            shape=new_shape,
            grad=self._grad,
            size=self.size,
            requires_grad=self.requires_grad
        )

        if self.requires_grad:
            out._prev = [self]
            def _backward():
                if out.grad is None:
                    return
                if self.grad is None:
                    self.grad = (c_float * self.size)()
                # for i in range(self.size):
                    # self.grad[i] += out.grad[i]
                SimdTensorBackend.tensor_add_inplace(self.grad, out.grad, self.size)
            out._backward = _backward

        return out

    def flatten(self, start_dim: int = 0) -> 'Tensor':
        """
        Flattens the tensor starting from `start_dim` into a single dimension.

        Args:
            start_dim (int): Dimension to start flattening from (default: 0).

        Returns:
            A new Tensor with flattened dimensions.

        Raises:
            ValueError: If start_dim is invalid.
        """
        if start_dim < 0 or start_dim >= len(self.shape):
            raise ValueError(f"Invalid start_dim: {start_dim}, tensor has {len(self.shape)} dimensions")

        pre_shape = self.shape[:start_dim]
        flatten_size = reduce(mul, self.shape[start_dim:], 1)
        new_shape = pre_shape + (flatten_size,) if pre_shape else (flatten_size,)

        out = Tensor(
            data=self.data,  # Share the data buffer
            shape=new_shape,
            grad=self._grad,  # Share the grad buffer
            size=self.size,
            requires_grad=self.requires_grad
        )

        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                if self.grad is None:
                    self.grad = (c_float * self.size)()
                # for i in range(self.size):
                    # self.grad[i] += out.grad[i]
                SimdTensorBackend.tensor_add_inplace(self.grad, out.grad, self.size)
            out._backward = _backward
            out._prev = [self]

        return out

    def conv2d(self, weight: 'Tensor', bias: Optional['Tensor'] = None, stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0)) -> 'Tensor':
        """
        Performs a 2D convolution operation.

        Args:
            weight: Weight tensor of shape (C_out, C_in, K_h, K_w).
            bias: Optional bias tensor of shape (C_out,).
            stride: Stride for height and width (default: (1, 1)).
            padding: Padding for height and width (default: (0, 0)).

        Returns:
            Output tensor after convolution.

        Raises:
            ValueError: If input or weight tensor shapes are invalid.
        """
        if self.ndim != 4:
            raise ValueError(f"Expected 4D input tensor, got shape {self.shape}")
        if weight.ndim != 4:
            raise ValueError(f"Expected 4D weight tensor, got shape {weight.shape}")

        N, C_in, H_in, W_in = self.shape
        C_out, C_weight, K_h, K_w = weight.shape
        if C_in != C_weight:
            raise ValueError(f"Input channel mismatch: expected C_in={C_in}, got C_weight={C_weight}")
        if bias is not None and bias.shape != (C_out,):
            raise ValueError(f"Expected bias shape ({C_out},), got {bias.shape}")

        stride_h, stride_w = stride
        pad_h, pad_w = padding

        H_out = (H_in + 2 * pad_h - K_h) // stride_h + 1
        W_out = (W_in + 2 * pad_w - K_w) // stride_w + 1

        out_size = N * C_out * H_out * W_out
        out_data = (c_float * out_size)()

        SimdTensorBackend.conv2d_forward_gemm(
            self.data,
            weight.data,
            bias.data if bias is not None else None,
            out_data,
            c_size_t(N), c_size_t(C_in), c_size_t(H_in), c_size_t(W_in),
            c_size_t(C_out), c_size_t(K_h), c_size_t(K_w),
            c_size_t(stride_h), c_size_t(stride_w),
            c_size_t(pad_h), c_size_t(pad_w),
        )

        out = Tensor(out_data, requires_grad=self.requires_grad or weight.requires_grad or (bias and bias.requires_grad),
                    shape=(N, C_out, H_out, W_out), size=out_size)

        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    raise RuntimeError("Output gradient is None before conv2d_backward")

                if self.requires_grad and self.grad is None:
                    self.grad = (c_float * self.size)()
                if weight.requires_grad and weight.grad is None:
                    weight.grad = (c_float * weight.size)()
                if bias and bias.requires_grad and bias.grad is None:
                    bias.grad = (c_float * bias.size)()

                SimdTensorBackend.conv2d_backward_gemm(
                    self.data,
                    weight.data,
                    out.grad,
                    self.grad if self.requires_grad else None,
                    weight.grad if weight.requires_grad else None,
                    bias.grad if bias and bias.requires_grad else None,
                    c_size_t(N), c_size_t(C_in), c_size_t(H_in), c_size_t(W_in),
                    c_size_t(C_out), c_size_t(K_h), c_size_t(K_w),
                    c_size_t(stride_h), c_size_t(stride_w),
                    c_size_t(pad_h), c_size_t(pad_w),
                )
            out._backward = _backward
            out._prev = [self, weight] + ([bias] if bias and bias.requires_grad else [])

        return out

    def avgpool2d(self, kernel_size: Union[int, Tuple[int, int]] = (2, 2), stride: Optional[Union[int, Tuple[int, int]]] = None) -> 'Tensor':
        """
        Applies 2D average pooling.

        Args:
            kernel_size: Size of the pooling window (int or tuple of (height, width)).
            stride: Stride for pooling (default: same as kernel_size).

        Returns:
            Pooled output tensor.
        """
        kernel_h, kernel_w = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        stride_h, stride_w = (kernel_h, kernel_w) if stride is None else (stride, stride) if isinstance(stride, int) else stride

        N, C, H, W = self.shape
        H_out = (H - kernel_h) // stride_h + 1
        W_out = (W - kernel_w) // stride_w + 1
        out_size = N * C * H_out * W_out
        out_data = (c_float * out_size)()

        SimdTensorBackend.avgpool2d_forward(
            self.data, out_data,
            c_size_t(N), c_size_t(C), c_size_t(H), c_size_t(W),
            c_size_t(kernel_h), c_size_t(kernel_w),
            c_size_t(stride_h), c_size_t(stride_w)
        )

        out = Tensor(out_data, shape=(N, C, H_out, W_out), size=out_size, requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                if self.grad is None:
                    self.grad = (c_float * (N * C * H * W))()
                SimdTensorBackend.avgpool2d_backward(
                    out.grad, self.grad,
                    c_size_t(N), c_size_t(C), c_size_t(H), c_size_t(W),
                    c_size_t(kernel_h), c_size_t(kernel_w),
                    c_size_t(stride_h), c_size_t(stride_w)
                )
            out._backward = _backward
            out._prev = [self]

        return out

    def maxpool2d(self, kernel_size: Union[int, Tuple[int, int]] = (2, 2), stride: Optional[Union[int, Tuple[int, int]]] = None) -> 'Tensor':
        """
        Applies 2D max pooling.

        Args:
            kernel_size: Size of the pooling window (int or tuple of (height, width)).
            stride: Stride for pooling (default: same as kernel_size).

        Returns:
            Pooled output tensor.

        Raises:
            ValueError: If output dimensions are invalid.
        """
        kernel_h, kernel_w = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        stride_h, stride_w = (kernel_h, kernel_w) if stride is None else (stride, stride) if isinstance(stride, int) else stride

        N, C, H, W = self.shape
        H_out = (H - kernel_h) // stride_h + 1
        W_out = (W - kernel_w) // stride_w + 1
        if H_out <= 0 or W_out <= 0:
            raise ValueError(f"Invalid output dimensions: H_out={H_out}, W_out={W_out}. "
                            f"Ensure input size ({H},{W}) is compatible with kernel ({kernel_h},{kernel_w}) and stride ({stride_h},{stride_w})")
        
        out_size = N * C * H_out * W_out
        out_data = (c_float * out_size)()

        SimdTensorBackend.maxpool2d_forward(
            self.data, out_data,
            c_size_t(N), c_size_t(C), c_size_t(H), c_size_t(W),
            c_size_t(kernel_h), c_size_t(kernel_w),
            c_size_t(stride_h), c_size_t(stride_w)
        )

        out = Tensor(out_data, shape=(N, C, H_out, W_out), size=out_size, requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    raise RuntimeError("Output gradient is None before maxpool2d_backward")
                if self.grad is None:
                    self.grad = (c_float * (N * C * H * W))()
                SimdTensorBackend.maxpool2d_backward(
                    self.data, out.grad, self.grad,
                    c_size_t(N), c_size_t(C), c_size_t(H), c_size_t(W),
                    c_size_t(kernel_h), c_size_t(kernel_w),
                    c_size_t(stride_h), c_size_t(stride_w)
                )
            out._backward = _backward
            out._prev = [self]

        return out

    def relu(self) -> 'Tensor':
        """
        Applies the ReLU activation function element-wise.

        Returns:
            Tensor with ReLU applied.
        """
        out_data = (c_float * self.size)()
        SimdTensorBackend.tensor_relu(self.data, out_data, self.size)
        out = Tensor(out_data, requires_grad=self.requires_grad, shape=self.shape, size=self.size)

        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    raise RuntimeError("Output gradient is None before relu_backward")
                if self.requires_grad:
                    tmp = (c_float * self.size)()
                    SimdTensorBackend.tensor_relu_backward(
                        out.grad, self.data, tmp, self.size
                    )
                    if self.grad is None:
                        self.grad = tmp
                    else:
                        SimdTensorBackend.tensor_add_inplace(self.grad, tmp, self.size)
            out._backward = _backward
            out._prev = [self]

        return out

    def sum(self) -> 'Tensor':
        out_data = (c_float * 1)()
        result = SimdTensorBackend.tensor_sum(self.data, None, self.size)
        out_data[0] = result
        out = Tensor(out_data, requires_grad=self.requires_grad, shape=(1,), size=1)
        if out.requires_grad:
            def _backward():
                if self.requires_grad and out.grad is not None:
                    scalar_grad = out.grad[0]
                    if self._grad is None:
                        # Allocate and fill self._grad with scalar_grad
                        self._grad = (c_float * self.size)(*[scalar_grad] * self.size)
                    else:
                        # Create a temp gradient buffer filled with scalar_grad
                        temp_grad = (c_float * self.size)(*[scalar_grad] * self.size)
                        # Accumulate into self._grad using optimized inplace add
                        SimdTensorBackend.tensor_add_inplace(self._grad, temp_grad, self.size)
            out._backward = _backward
            out._prev = [self]

        return out

    def mean(self) -> 'Tensor':
        """
        Computes the mean of all elements in the tensor.

        Returns:
            Scalar tensor containing the mean.
        """
        out_data = (c_float * 1)()
        result = SimdTensorBackend.tensor_mean(self.data, None, self.size)
        out_data[0] = result
        out = Tensor(out_data, requires_grad=self.requires_grad, shape=(1,), size=1)

        if out.requires_grad:
            def _backward():
                if self.requires_grad and out.grad is not None:
                    scalar_grad = out.grad[0] / self.size
                    if self._grad is None:
                        # First time: allocate and fill with scalar_grad
                        self._grad = (c_float * self.size)(*[scalar_grad] * self.size)
                    else:
                        # Create a temp gradient buffer filled with scalar_grad
                        temp_grad = (c_float * self.size)(*[scalar_grad] * self.size)
                        # Accumulate into self._grad
                        SimdTensorBackend.tensor_add_inplace(self._grad, temp_grad, self.size)
            out._backward = _backward
            out._prev = [self]

        return out

    def cross_entropy(self, target: 'Tensor', label_smoothing: float = 0.0, use_label_smoothing: int = 0) -> 'Tensor':
        """
        Computes cross-entropy loss between the tensor and target indices.

        Args:
            target: Tensor of class indices with shape (batch_size,).
            label_smoothing: Smoothing factor for labels (default: 0.0).
            use_label_smoothing: Whether to apply label smoothing (default: 0).

        Returns:
            Scalar tensor containing the mean cross-entropy loss.

        Raises:
            ValueError: If target shape or indices are invalid.
        """
        B, C = self.shape
        if target.shape != (B,):
            raise ValueError(f"cross_entropy() expects target to be class indices of shape ({B},), got {target.shape}")
        
        target_data = [int(target.data[i]) for i in range(B)]
        if any(t < 0 or t >= C for t in target_data):
            raise ValueError(f"Target indices must be in range [0, {C-1}], got {target_data}")
        
        target_int = (c_int * B)(*target_data)
        target_ptr = cast(target_int, POINTER(c_int))

        loss_data = (c_float * B)()
        grad_input = (c_float * (B * C))()
        probs_data = (c_float * (B * C))()

        SimdTensorBackend.tensor_softmax_ce(
            self.data,
            target_ptr,
            None,             # No incoming grad for forward pass
            loss_data,
            grad_input,
            probs_data,
            B,
            C,
            c_float(label_smoothing),
            c_int(use_label_smoothing)
        )

        # Per-sample loss
        loss = Tensor(loss_data, requires_grad=self.requires_grad, shape=(B,), size=B)

        if loss.requires_grad:
            loss._prev = [self]
            def _backward():
                if loss.grad is None:
                    raise RuntimeError("Loss grad is None before softmax_ce backward")
                if self.grad is None:
                    self.grad = (c_float * (B * C))()
                tmp = (c_float * (B * C))()
                SimdTensorBackend.tensor_softmax_ce(
                    self.data,
                    target_ptr,
                    loss.grad,
                    loss_data,
                    tmp,
                    None,
                    B,
                    C,
                    c_float(label_smoothing),
                    c_int(use_label_smoothing)
                )
                for i in range(B * C):
                    self.grad[i] += tmp[i]
            loss._backward = _backward

        # Mean loss *after* backward graph is set up
        loss_mean = loss.mean()
        loss_mean._prev = [loss]  # Link mean to loss for backward
        return loss_mean

    def matmul(self, other: 'Tensor') -> 'Tensor':
        """
        Performs matrix multiplication with another tensor, supporting PyTorch-like broadcasting.

        Args:
            other: Tensor to multiply with.

        Returns:
            Result of matrix multiplication.

        Raises:
            ValueError: If shapes are incompatible.
            NotImplementedError: If shapes are unsupported.
        """
        if not isinstance(other, Tensor):
            raise ValueError("Operand must be a Tensor")

        s1, s2 = self.shape, other.shape
        d1, d2 = len(s1), len(s2)

        # Handle scalar-like cases (e.g., (1,) @ (1,))
        if d1 == 1 and d2 == 1 and s1[0] == 1 and s2[0] == 1:
            out_data = (c_float * 1)(self.data[0] * other.data[0])
            return Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad, shape=(), size=1)

        # Handle 1D cases (dot product, vector-matrix, matrix-vector)
        if d1 == 1 and d2 == 1:
            if s1[0] != s2[0]:
                raise ValueError(f"Incompatible dot product shapes {s1} and {s2}")
            batch, M, K, N = 1, 1, s1[0], 1
            out_shape = ()
        elif d1 == 1 and d2 == 2:
            if s1[0] != s2[0]:
                raise ValueError(f"Incompatible shapes {s1} and {s2}")
            batch, M, K, N = 1, 1, s1[0], s2[1]
            out_shape = (N,)
        elif d1 == 2 and d2 == 1:
            if s1[1] != s2[0]:
                raise ValueError(f"Incompatible shapes {s1} and {s2}")
            batch, M, K, N = 1, s1[0], s1[1], 1
            out_shape = (M,)
        else:
            # Normalize to at least 2D
            s1 = (1,) * (max(2, d2) - d1) + s1
            s2 = (1,) * (max(2, d1) - d2) + s2
            d1, d2 = len(s1), len(s2)

            # Validate matrix dimensions
            M, K = s1[-2], s1[-1]
            K2, N = s2[-2], s2[-1]
            if K != K2:
                raise ValueError(f"Incompatible matrix dimensions {s1}[-2:]={s1[-2:]} and {s2}[-2:]={s2[-2:]}")

            # Broadcast batch dimensions
            batch_dims = []
            for b1, b2 in zip(s1[:-2], s2[:-2]):
                if b1 == 1:
                    batch_dims.append(b2)
                elif b2 == 1:
                    batch_dims.append(b1)
                elif b1 == b2:
                    batch_dims.append(b1)
                else:
                    raise ValueError(f"Incompatible batch dimensions {s1[:-2]} and {s2[:-2]}")
            batch = 1 if not batch_dims else batch_dims[0] if len(batch_dims) == 1 else tuple(batch_dims)
            out_shape = batch + (M, N) if isinstance(batch, tuple) else (batch, M, N) if batch != 1 else (M, N)

        # Calculate output size
        out_size = max(1, M * N * (batch if isinstance(batch, int) else batch[0] if batch else 1))
        out_data = (c_float * out_size)()

        # Reshape inputs for backend
        self_shape = batch + (M, K) if isinstance(batch, tuple) else (batch, M, K) if batch != 1 else (M, K)
        other_shape = batch + (K, N) if isinstance(batch, tuple) else (batch, K, N) if batch != 1 else (K, N)
        self_reshaped = self.reshape(self_shape)
        other_reshaped = other.reshape(other_shape)

        # Check for NaNs in inputs
        for i in range(self_reshaped.size):
            if not (-1e10 < self_reshaped.data[i] < 1e10):
                print(f"Warning: NaN or extreme value in self.data at index {i}: {self_reshaped.data[i]}")
        for i in range(other_reshaped.size):
            if not (-1e10 < other_reshaped.data[i] < 1e10):
                print(f"Warning: NaN or extreme value in other.data at index {i}: {other_reshaped.data[i]}")

        # Perform matrix multiplication
        SimdTensorBackend.matmul_forward(
            self_reshaped.data,
            other_reshaped.data,
            out_data,
            batch if isinstance(batch, int) else batch[0], M, K, N
        )

        # Check output for NaNs
        for i in range(out_size):
            if not (-1e10 < out_data[i] < 1e10):
                print(f"Warning: NaN or extreme value in out_data at index {i}: {out_data[i]}")

        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad, shape=out_shape, size=out_size)

        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    raise RuntimeError("Output gradient is None before matmul_backward")
                if self.requires_grad and self.grad is None:
                    self.grad = (c_float * self.size)()
                if other.requires_grad and other.grad is None:
                    other.grad = (c_float * other.size)()

                SimdTensorBackend.matmul_backward(
                    self_reshaped.data,
                    other_reshaped.data,
                    out.grad,
                    self.grad if self.requires_grad else None,
                    other.grad if other.requires_grad else None,
                    batch if isinstance(batch, int) else batch[0], M, K, N,
                    True  # Accumulate gradients
                )

                # Check gradients for NaNs
                if self.requires_grad:
                    for i in range(self.size):
                        if not (-1e10 < self.grad[i] < 1e10):
                            print(f"Warning: NaN or extreme value in self.grad at index {i}: {self.grad[i]}")
                if other.requires_grad:
                    for i in range(other.size):
                        if not (-1e10 < other.grad[i] < 1e10):
                            print(f"Warning: NaN or extreme value in other.grad at index {i}: {other.grad[i]}")
            out._backward = _backward
            out._prev = [self, other]

        return out
    
    def clone(self):
        new_data = (c_float * self.size)(*[self.data[i] for i in range(self.size)])
        out = Tensor(new_data, requires_grad=self.requires_grad, grad=None, shape=self.shape, size=self.size)
        out._prev = []
        out._backward = None
        out.owns_data = True
        return out

    def detach(self):
        out = Tensor(data=self.data, requires_grad=False, shape=self.shape,size=self.size,grad=None)
        out._backward = None
        out._prev = []
        out._grad = None
        return out
        
    def release(self) -> None:
        """
        Frees computation-related metadata and data buffer (if not a parameter).
        """
        self._backward = None
        self._prev = []
        self._grad = None
        if not getattr(self, "_is_param", False):
            self.data = None

    def release_graph(self) -> None:
        visited: Set['Tensor'] = set()
        def _recurse(t: 'Tensor') -> None:
            if t in visited:
                return
            visited.add(t)
            for p in getattr(t, "_prev", []):
                _recurse(p)
            t._backward = None
            t._prev = []
            t._grad = None
            if getattr(t, "_release_data", False):
                t.data = None
            for attr in ['_mask', '_cached_a_broadcasted', '_cached_b_broadcasted', '_batch_info']:
                if hasattr(t, attr):
                    delattr(t, attr)
        _recurse(self)

    def permute(self, *axes) -> 'Tensor':
        """
        Permutes the dimensions of the tensor according to the specified axes.

        Args:
            *axes: The desired ordering of dimensions (e.g., (1, 0, 2) for a 3D tensor).

        Returns:
            Tensor: A new tensor with permuted dimensions.

        Raises:
            ValueError: If axes are invalid or incompatible with tensor shape.
        """
        if len(axes) != len(self.shape):
            raise ValueError(f"Expected {len(self.shape)} axes, got {len(axes)}")
        if sorted(axes) != list(range(len(self.shape))):
            raise ValueError(f"Invalid axes {axes} for shape {self.shape}")

        # Compute new shape
        new_shape = tuple(self.shape[i] for i in axes)
        out_size = self.size
        out_data = (c_float * out_size)()

        # Compute strides for input and output
        in_strides = [1] * len(self.shape)
        for i in range(len(self.shape) - 2, -1, -1):
            in_strides[i] = in_strides[i + 1] * self.shape[i + 1]
        
        out_strides = [1] * len(new_shape)
        for i in range(len(new_shape) - 2, -1, -1):
            out_strides[i] = out_strides[i + 1] * new_shape[i + 1]

        # Map indices
        for idx in range(out_size):
            # Convert flat index to multi-dimensional index in output
            out_multi_idx = [0] * len(new_shape)
            tmp = idx
            for i in range(len(new_shape) - 1, -1, -1):
                out_multi_idx[i] = tmp // out_strides[i]
                tmp %= out_strides[i]
            
            # Map to input multi-dimensional index using axes
            in_multi_idx = [0] * len(self.shape)
            for i, ax in enumerate(axes):
                in_multi_idx[ax] = out_multi_idx[i]
            
            # Convert input multi-dimensional index to flat index
            in_idx = 0
            for i in range(len(self.shape)):
                in_idx += in_multi_idx[i] * in_strides[i]
            
            out_data[idx] = self.data[in_idx]

        out = Tensor(out_data, requires_grad=self.requires_grad, shape=new_shape, size=out_size)

        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    raise RuntimeError("Output gradient is None during permute backward")
                if self.grad is None:
                    self.grad = (c_float * self.size)()

                # Inverse permutation: map output gradient back to input
                for idx in range(out_size):
                    out_multi_idx = [0] * len(new_shape)
                    tmp = idx
                    for i in range(len(new_shape) - 1, -1, -1):
                        out_multi_idx[i] = tmp // out_strides[i]
                        tmp %= out_strides[i]
                    
                    in_multi_idx = [0] * len(self.shape)
                    for i, ax in enumerate(axes):
                        in_multi_idx[ax] = out_multi_idx[i]
                    
                    in_idx = 0
                    for i in range(len(self.shape)):
                        in_idx += in_multi_idx[i] * in_strides[i]
                    
                    self.grad[in_idx] += out.grad[idx]

            out._backward = _backward
            out._prev = [self]

        return out

    def log_softmax(self, dim: int = -1) -> 'Tensor':
        """
        Computes the log softmax along the specified dimension.

        Args:
            dim (int): Dimension along which to compute log softmax (default: -1, last dimension).

        Returns:
            Tensor: Log softmax of the input tensor along the specified dimension.

        Raises:
            ValueError: If dim is invalid or tensor is empty.
        """
        if not self.shape:
            raise ValueError("Cannot compute log_softmax on empty tensor")
        if dim >= len(self.shape) or dim < -len(self.shape):
            raise ValueError(f"Dimension {dim} out of bounds for shape {self.shape}")

        # Normalize negative dimension
        dim = dim if dim >= 0 else dim + len(self.shape)

        # Permute if dim is not the last dimension
        need_permute = dim != len(self.shape) - 1
        if need_permute:
            axes = list(range(len(self.shape)))
            axes[-1], axes[dim] = axes[dim], axes[-1]
            x = self.permute(*axes)
        else:
            x = self

        input_data = x.data
        input_shape = x.shape
        batch_size = reduce(mul, input_shape[:-1], 1) if len(input_shape) > 1 else 1
        class_size = input_shape[-1]
        out_data = (c_float * x.size)()

        # Compute log softmax with numerical stability
        for b in range(batch_size):
            start = b * class_size
            max_val = max(input_data[start:start + class_size])
            exp_sum = sum(math.exp(input_data[start + i] - max_val) for i in range(class_size))
            log_sum = max_val + math.log(exp_sum) if exp_sum > 0 else float('-inf')
            for i in range(class_size):
                out_data[start + i] = input_data[start + i] - log_sum

        out = Tensor(out_data, requires_grad=self.requires_grad, shape=x.shape, size=x.size)

        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    raise RuntimeError("Output gradient is None during log_softmax backward")
                if x.grad is None:
                    x.grad = (c_float * x.size)()

                # Compute softmax and gradient
                for b in range(batch_size):
                    start = b * class_size
                    max_val = max(input_data[start:start + class_size])
                    exp_sum = sum(math.exp(input_data[start + i] - max_val) for i in range(class_size))
                    softmax = [(math.exp(input_data[start + i] - max_val) / exp_sum if exp_sum > 0 else 0.0)
                            for i in range(class_size)]
                    grad_sum = sum(out.grad[start + i] for i in range(class_size))
                    for i in range(class_size):
                        x.grad[start + i] = out.grad[start + i] - softmax[i] * grad_sum

                # Handle gradient permutation if needed
                if need_permute:
                    grad_tensor = Tensor(x.grad, shape=x.shape, size=x.size, requires_grad=False)
                    grad_tensor = grad_tensor.permute(*axes)
                    if self.grad is None:
                        self.grad = (c_float * self.size)()
                    SimdTensorBackend.tensor_add_inplace(self.grad, grad_tensor.data, self.size)
                else:
                    if self.grad is None:
                        self.grad = (c_float * self.size)()
                    SimdTensorBackend.tensor_add_inplace(self.grad, x.grad, self.size)

            out._backward = _backward
            out._prev = [self]

        # Permute output back to original shape if needed
        if need_permute:
            out = out.permute(*axes)

        return out

    def tanh(self) -> 'Tensor':
        """
        Applies the hyperbolic tangent (tanh) activation function element-wise.

        Returns:
            Tensor with tanh applied.

        Notes:
            The gradient of tanh(x) is 1 - tanh(x)^2.
        """
        out_data = (c_float * self.size)()
        for i in range(self.size):
            out_data[i] = math.tanh(self.data[i])
        out = Tensor(out_data, requires_grad=self.requires_grad, shape=self.shape, size=self.size)

        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    raise RuntimeError("Output gradient is None during tanh backward")
                if self.grad is None:
                    self.grad = (c_float * self.size)()
                tmp = (c_float * self.size)()
                for i in range(self.size):
                    tanh_x = out.data[i]
                    tmp[i] = out.grad[i] * (1.0 - tanh_x * tanh_x)
                SimdTensorBackend.tensor_add_inplace(self.grad, tmp, self.size)

            out._backward = _backward
            out._prev = [self]

        return out

    def dropout(self, p: float) -> 'Tensor':
        """
        Applies dropout with probability p during training.

        Args:
            p: Probability of dropping an element (0 <= p < 1).

        Returns:
            Tensor with dropout applied.
        """
        if not self.requires_grad or p <= 0:
            return self
        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")

        scale = 1.0 / (1.0 - p)
        size = self.size
        shape = self.shape
        out_data = (c_float * size)()
        mask_array = (c_float * size)()

        SimdTensorBackend.tensor_dropout(self.data, out_data, mask_array, size, p, scale)

        out = Tensor(out_data, requires_grad=self.requires_grad, shape=shape, size=size)

        if out.requires_grad:
            out._mask = mask_array
            def _backward():
                if out.grad is None:
                    return
                if self._grad is None:
                    self._grad = (c_float * self.size)()
                SimdTensorBackend.tensor_mul(out.grad, out._mask, self._grad, self.size, 0)
                out._mask = None
            
            out._backward = _backward
            out._prev = [self]
            
        return out

    def backward(self) -> None:
        """
        Computes gradients via reverse-mode autodiff.

        Assumes this tensor is a scalar output of a computation graph.
        Raises an error if not scalar (unless .sum() was used).
        """
        if self.shape != (1,) and not (len(self.shape) == 0 or self.shape == ()):
            raise RuntimeError(
                f"Cannot call backward on non-scalar tensor with shape {self.shape}. "
                "Call `.sum().backward()` or pass an explicit gradient instead."
            )

        # Topological sort of computation graph
        topo: list['Tensor'] = []
        visited: set['Tensor'] = set()
        def build_topo(t: 'Tensor'):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)
        build_topo(self)

        # Initialize self.grad as 1.0 (for scalar output)
        if self.grad is None:
            self.grad = (c_float * self.size)()
        SimdTensorBackend.tensor_fill_inplace(self.grad, c_float(1.0), c_size_t(self.size))

        # Backward pass: reversed topological order
        for t in reversed(topo):
            if t._backward is not None:
                try:
                    t._backward()
                except Exception as e:
                    raise RuntimeError(f"Error during backward pass for tensor with shape {t.shape}: {e}")
                t._backward = None  # release memory

    def __repr__(self) -> str:
        """String representation of the tensor."""
        data_list = [self.data[i] for i in range(min(self.size, 10))]
        grad_list = [self.grad[i] for i in range(min(self.size, 10))] if self.grad else None
        return f"Tensor(shape={self.shape}, data={data_list}{'...' if self.size > 10 else ''}, grad={grad_list}{'...' if self.grad and self.size > 10 else ''})"

def zeros(shape: Union[int, Tuple[int, ...]], requires_grad: bool = False) -> Tensor:
    """
    Creates a tensor filled with zeros.

    Args:
        shape: Shape of the tensor (int or tuple of ints).
        requires_grad: If True, tracks gradients (default: False).

    Returns:
        Tensor filled with zeros.
    """
    shape = (shape,) if isinstance(shape, int) else shape
    size = reduce(mul, shape, 1)
    data = (c_float * size)(0.0)
    return Tensor(data, requires_grad=requires_grad, shape=shape, size=size)

def ones(shape: Union[int, Tuple[int, ...]], requires_grad: bool = False) -> Tensor:
    """
    Creates a tensor filled with ones.

    Args:
        shape: Shape of the tensor (int or tuple of ints).
        requires_grad: If True, tracks gradients (default: False).

    Returns:
        Tensor filled with ones.
    """
    shape = (shape,) if isinstance(shape, int) else shape
    size = reduce(mul, shape, 1)
    data = (c_float * size)(*[1.0] * size)
    return Tensor(data, requires_grad=requires_grad, shape=shape, size=size)

def rand(shape: Union[int, Tuple[int, ...]], requires_grad: bool = False) -> Tensor:
    """
    Creates a tensor filled with random values in [0, 1).

    Args:
        shape: Shape of the tensor (int or tuple of ints).
        requires_grad: If True, tracks gradients (default: False).

    Returns:
        Tensor filled with random values.
    """
    shape = (shape,) if isinstance(shape, int) else shape
    size = reduce(mul, shape, 1)
    data = (c_float * size)()
    for i in range(size):
        data[i] = random.random()
    return Tensor(data, requires_grad=requires_grad, shape=shape, size=size)

def randn(shape: Union[int, Tuple[int, ...]], mean: float = 0.0, std: float = 1.0, requires_grad: bool = False) -> Tensor:
    """
    Creates a tensor filled with random values from a normal distribution.

    Args:
        shape: Shape of the tensor (int or tuple of ints).
        mean: Mean of the normal distribution (default: 0.0).
        std: Standard deviation of the normal distribution (default: 1.0).
        requires_grad: If True, tracks gradients (default: False).

    Returns:
        Tensor filled with random normal values.
    """
    shape = (shape,) if isinstance(shape, int) else shape
    size = reduce(mul, shape, 1)
    data = (c_float * size)()
    for i in range(size):
        # Box-Muller transform for normal distribution
        u1, u2 = random.random(), random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        data[i] = mean + std * z
    return Tensor(data, requires_grad=requires_grad, shape=shape, size=size)

def from_ctypes(ptr, shape, size, requires_grad=False):
    t = Tensor.__new__(Tensor)
    t.data = ptr
    t.shape = shape
    t.size = size
    t.requires_grad = requires_grad
    t.grad = None
    t._grad = None
    t._prev = []
    t._backward = None
    t.owns_data = False
    return t

Tensor.zeros = staticmethod(zeros)
Tensor.randn = staticmethod(randn)
Tensor.from_ctypes = staticmethod(from_ctypes)

def _init_backend():
    result = SimdTensorBackend.tensor_ops_init()
    if result != 0:
        raise RuntimeError("Failed to initialize SIMD backend (AVX2 unsupported?)")
    
_init_backend()