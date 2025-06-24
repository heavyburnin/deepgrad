from deepgrad.backend import SimdTensorBackend
from ctypes import c_float, c_size_t, Array
from deepgrad.broadcast import compute_broadcast_shape, broadcast_to_shape, unbroadcast_grad
from deepgrad.ops import get_op_names

class Tensor:
    def __init__(self, data, requires_grad=False, shape=None, size=None):
        self.data = data
        self.requires_grad = requires_grad
        self._grad = None
        self._backward = None
        self._prev = []

        if shape is not None:
            self.shape = shape
        else:
            # Infer 1D shape if not given
            if size is not None:
                self.shape = (size,)
            else:
                self.shape = (len(data),)

        # Compute expected size from shape
        expected_size = 1
        for dim in self.shape:
            expected_size *= dim

        # Determine actual size
        if size is not None:
            self.size = size
        else:
            try:
                self.size = len(data)
            except TypeError:
                raise ValueError("Must provide `size=` when using raw ctypes memory")

        # Validate size matches shape
        assert self.size == expected_size, f"Shape {self.shape} incompatible with data size {self.size}"

    def __getstate__(self):
        state = self.__dict__.copy()

        # Convert ctypes data buffer to list
        if 'data' in state and hasattr(state['data'], '__len__') and isinstance(state['data'], Array):
            state['data'] = [state['data'][i] for i in range(len(state['data']))]

        # Convert ctypes grad buffer to list
        if '_grad' in state and state['_grad'] is not None:
            if hasattr(state['_grad'], '__len__') and isinstance(state['_grad'], Array):
                state['_grad'] = [state['_grad'][i] for i in range(len(state['_grad']))]

        return state

    def __setstate__(self, state):
        # Convert list back to ctypes array for data
        if 'data' in state and isinstance(state['data'], list):
            state['data'] = (c_float * len(state['data']))(*state['data'])

        # Convert list back to ctypes array for grad
        if '_grad' in state and isinstance(state['_grad'], list):
            state['_grad'] = (c_float * len(state['_grad']))(*state['_grad'])

        self.__dict__.update(state)

    def __len__(self):
        return self.shape[0] if self.shape else 1

    @property
    def grad(self):
        if not self.requires_grad:
            return None

        if self._grad is None:
            self._grad = (c_float * self.size)()  # freshly allocated zero buffer
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    @property
    def ndim(self):
        return len(self.shape)

    def _apply_op(self, other, op_name, grad_fn_name):
        if not isinstance(other, Tensor):
            other = Tensor([other], shape=(1,), requires_grad=False)

        # Determine output shape
        out_shape = compute_broadcast_shape(self.shape, other.shape)  # keep it

        # Broadcast data
        a_broadcasted = (
            self.data if self.shape == out_shape 
            else broadcast_to_shape(self.data, self.shape, out_shape, self.size)
        )

        b_broadcasted = (
            other.data if other.shape == out_shape 
            else broadcast_to_shape(other.data, other.shape, out_shape, other.size)
        )

        # Prepare output buffer
        out_size = 1
        for dim in out_shape:
            out_size *= dim
        out_data = (c_float * out_size)()

        # Compute batching
        if len(out_shape) > 1:
            batch_size = out_shape[0]
            n = 1
            for dim in out_shape[1:]:
                n *= dim
            use_batch = True
        else:
            batch_size = 1
            n = out_size
            use_batch = False

        # Call C op
        getattr(SimdTensorBackend, op_name)(
            a_broadcasted,
            b_broadcasted,
            out_data,
            n,
            batch_size if use_batch else 0  # only passed if batched
        )

        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad, shape=out_shape)

        # Cache broadcasted arrays only if needed
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
                    grad_fn(out_grad, a_broadcasted, b_broadcasted, self_grad, other_grad, n)

                if self.requires_grad:
                    self.grad = unbroadcast_grad(self_grad, out.shape, self.shape)

                if other.requires_grad:
                    other.grad = unbroadcast_grad(other_grad, out.shape, other.shape)

            out._backward = _backward
            out._prev = [self, other]

        return out
    
    def _binary_op(self, other, op_name):
        forward_fn, backward_fn = get_op_names(op_name)
        return self._apply_op(other, forward_fn, backward_fn)
    
    def __add__(self, other): return self._binary_op(other, 'add')
    def __radd__(self, other): return self.__add__(other)

    def __sub__(self, other): return self._binary_op(other, 'sub')
    def __rsub__(self, other): return Tensor(other, requires_grad=False).__sub__(self)

    def __mul__(self, other): return self._binary_op(other, 'mul')
    def __rmul__(self, other): return self.__mul__(other)

    def __truediv__(self, other): return self._binary_op(other, 'div')
    def __rtruediv__(self, other): return Tensor(other, requires_grad=False).__truediv__(self)

    def __pow__(self, other): return self._binary_op(other, 'pow')
    def __rpow__(self, other): return Tensor(other, requires_grad=False).__pow__(self)

    def reshape(self, new_shape):
        from functools import reduce
        from operator import mul

        inferred = -1
        known_product = 1
        inferred_index = -1
        for i, dim in enumerate(new_shape):
            if dim == -1:
                assert inferred_index == -1, "Only one dimension can be inferred"
                inferred_index = i
            else:
                known_product *= dim

        original_product = reduce(mul, self.shape, 1)
        if inferred_index != -1:
            assert original_product % known_product == 0, "Inferred dimension must divide total size"
            inferred = original_product // known_product
            new_shape = list(new_shape)
            new_shape[inferred_index] = inferred

        assert reduce(mul, new_shape, 1) == original_product, "Reshape must preserve total size"

        return Tensor(self.data, shape=tuple(new_shape), size=original_product, requires_grad=self.requires_grad)

    def conv2d(self, weight, bias=None, stride=(1, 1), padding=(0, 0)):
        assert self.ndim == 4, f"Expected 4D input tensor, got shape {self.shape}"
        assert weight.ndim == 4, f"Expected 4D weight tensor, got shape {weight.shape}"

        N, C_in, H_in, W_in = self.shape
        C_out, C_weight, K_h, K_w = weight.shape
        assert C_in == C_weight, "Input channel mismatch between input and weight"

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

        out = Tensor(out_data, requires_grad=self.requires_grad or weight.requires_grad or (bias and bias.requires_grad), shape=(N, C_out, H_out, W_out), size=out_size)

        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return

                grad_input = self.grad if self.requires_grad else None
                grad_weight = weight.grad if weight.requires_grad else None
                grad_bias = bias.grad if bias and bias.requires_grad else None

                SimdTensorBackend.conv2d_backward_gemm(
                    self.data,
                    weight.data,
                    out.grad,
                    grad_input,
                    grad_weight,
                    grad_bias,
                    c_size_t(N), c_size_t(C_in), c_size_t(H_in), c_size_t(W_in),
                    c_size_t(C_out), c_size_t(K_h), c_size_t(K_w),
                    c_size_t(stride_h), c_size_t(stride_w),
                    c_size_t(pad_h), c_size_t(pad_w),
                )

            out._backward = _backward
            out._prev = [self, weight] + ([bias] if bias and bias.requires_grad else [])

        return out

    def avgpool2d(self, kernel_size=(2, 2), stride=None):
        # --- Normalize kernel size ---
        if isinstance(kernel_size, int):
            kernel_h = kernel_w = kernel_size
        else:
            kernel_h, kernel_w = kernel_size

        # --- Normalize stride ---
        if stride is None:
            stride_h, stride_w = kernel_h, kernel_w
        elif isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        # --- Compute output shape ---
        N, C, H, W = self.shape
        H_out = (H - kernel_h) // stride_h + 1
        W_out = (W - kernel_w) // stride_w + 1
        out_size = N * C * H_out * W_out
        out_data = (c_float * out_size)()

        # --- Forward ---
        SimdTensorBackend.avgpool2d_forward(
            self.data, out_data,
            c_size_t(N), c_size_t(C), c_size_t(H), c_size_t(W),
            c_size_t(kernel_h), c_size_t(kernel_w),
            c_size_t(stride_h), c_size_t(stride_w)
        )

        out = Tensor(out_data, shape=(N, C, H_out, W_out), size=out_size, requires_grad=self.requires_grad)

        # --- Backward ---
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

    def maxpool2d(self, kernel_size=(2, 2), stride=None):
        # --- Normalize kernel size ---
        if isinstance(kernel_size, int):
            kernel_h = kernel_w = kernel_size
        else:
            kernel_h, kernel_w = kernel_size

        # --- Normalize stride ---
        if stride is None:
            stride_h, stride_w = kernel_h, kernel_w
        elif isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        # --- Shape info ---
        N, C, H, W = self.shape
        H_out = (H - kernel_h + stride_h) // stride_h
        W_out = (W - kernel_w + stride_w) // stride_w
        out_size = N * C * H_out * W_out

        out_data = (c_float * out_size)()

        # --- Forward ---
        SimdTensorBackend.maxpool2d_forward(
            self.data, out_data,
            c_size_t(N), c_size_t(C), c_size_t(H), c_size_t(W),
            c_size_t(kernel_h), c_size_t(kernel_w),
            c_size_t(stride_h), c_size_t(stride_w)
        )

        out = Tensor(out_data, shape=(N, C, H_out, W_out), size=out_size, requires_grad=self.requires_grad)

        # --- Backward ---
        if self.requires_grad:
            def _backward():
                if out.grad is None:
                    return
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

    def relu(self):
        out_data = (c_float * self.size)()

        SimdTensorBackend.tensor_relu(self.data, out_data, self.size)

        out = Tensor(out_data, requires_grad=self.requires_grad, shape=self.shape, size=self.size)

        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                if self.requires_grad:
                    SimdTensorBackend.tensor_relu_backward(
                        out.grad,
                        self.data,
                        self.grad,
                        self.size
                    )
            out._backward = _backward
            out._prev = [self]

        return out
    
    def cross_entropy(self, target):
        assert self.shape == target.shape, f"Shape mismatch: {self.shape} vs {target.shape}"
        B, C = self.shape
        loss_data = (c_float * B)()
        grad_input = (c_float * (B * C))()
        probs_data = (c_float * (B * C))()

        SimdTensorBackend.tensor_softmax_ce(
            self.data,
            target.data,
            None,
            loss_data,
            grad_input,
            probs_data,
            B,
            C
        )

        loss = Tensor(loss_data, requires_grad=self.requires_grad, shape=(B,), size=B)

        if loss.requires_grad:
            def _backward():
                if self.requires_grad:
                    SimdTensorBackend.tensor_softmax_ce(
                        self.data,
                        target.data,
                        loss.grad,
                        loss_data,
                        self.grad,
                        None,
                        B,
                        C
                    )
            loss._backward = _backward
            loss._prev = [self, target]

        return loss.mean()

    def matmul(self, other):
        assert isinstance(other, Tensor), "Operand must be a Tensor"

        s1, s2 = self.shape, other.shape

        # Determine dimensions and batching
        if len(s1) == 2 and len(s2) == 2:
            M, K = s1
            K2, N = s2
            assert K == K2, f"Incompatible matmul shapes {s1} and {s2}"
            batch = 1
        elif len(s1) == 3 and len(s2) == 2:
            B, M, K = s1
            K2, N = s2
            assert K == K2, f"Incompatible matmul shapes {s1} and {s2}"
            batch = B
        elif len(s1) == 2 and len(s2) == 3:
            M, K = s1
            B, K2, N = s2
            assert K == K2, f"Incompatible matmul shapes {s1} and {s2}"
            batch = B
        else:
            raise NotImplementedError(f"Unsupported shapes for matmul: {s1} @ {s2}")

        # Output shape/size
        out_shape = (batch, M, N) if batch > 1 else (M, N)
        out_size = batch * M * N if batch > 1 else M * N

        out_data = (c_float * out_size)()

        # Forward pass
        SimdTensorBackend.tensor_matmul(
            0,  # MATMUL_FORWARD
            self.data,
            other.data,
            None,           # grad_out (unused)
            out_data,
            None,           # grad_B (unused)
            batch,
            M, K, N,
            False
        )

        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad, shape=out_shape, size=out_size)

        # Backward setup
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return

                grad_A = self.grad if self.requires_grad else None
                grad_B = other.grad if other.requires_grad else None

                SimdTensorBackend.tensor_matmul(
                    1,  # MATMUL_BACKWARD
                    self.data,
                    other.data,
                    out.grad,
                    grad_A,
                    grad_B,
                    batch,
                    M, K, N,
                    True
                )

            out._backward = _backward
            out._prev = [self, other]

        return out

    def sum(self):
        out_data = (c_float * 1)()
        result = SimdTensorBackend.tensor_sum(self.data, None, self.size)
        out_data[0] = result
        out = Tensor(out_data, requires_grad=self.requires_grad)

        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    SimdTensorBackend.tensor_sum(self.data, self.grad, self.size)
            out._backward = _backward
            out._prev = [self]

        return out

    def mean(self):
        out_data = (c_float * 1)()
        result = SimdTensorBackend.tensor_mean(self.data, None, self.size)
        out_data[0] = result
        out = Tensor(out_data, requires_grad=self.requires_grad)

        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    SimdTensorBackend.tensor_mean(self.data, self.grad, self.size)
            out._backward = _backward
            out._prev = [self]

        return out

    def backward(self):
        visited = set()
        topo = []

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for p in t._prev:
                    build_topo(p)
                topo.append(t)

        build_topo(self)

        for t in topo:
            if t.requires_grad:
                if t._grad is None:
                    t._grad = (c_float * t.size)()
                else:
                    SimdTensorBackend.zero_float_array(t._grad, t.size)

        if self.shape != (1,) and not (len(self.shape) == 0 or self.shape == ()): 
            raise RuntimeError(
                f"Cannot call backward on non-scalar tensor with shape {self.shape}. "
                "Call `.sum().backward()` or pass an explicit gradient instead."
            )

        SimdTensorBackend.tensor_fill_inplace(self.grad, c_float(1.0), c_size_t(len(self.grad)))

        for t in reversed(topo):
            if t._backward is not None:
                t._backward()

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={[self.data[i] for i in range(len(self.data))]}, grad={[self.grad[i] for i in range(len(self.grad))] if self.grad else None})"

def _init_backend():
    result = SimdTensorBackend.tensor_ops_init()
    if result != 0:
        raise RuntimeError("tensor_ops_init failed (AVX2 unsupported?)")
    
_init_backend()