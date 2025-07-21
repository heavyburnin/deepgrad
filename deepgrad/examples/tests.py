import unittest
import numpy as np
from ctypes import c_float
from deepgrad.tensor import Tensor
from deepgrad.broadcast import broadcast_to_shape
import pickle


# Constants for gradient checking
EPS = 1e-4  # For finite difference
TOL = 1e-2  # Acceptable relative error for gradients

def to_numpy(tensor):
    """Convert Tensor data to a NumPy array with the same shape."""
    return np.array([tensor.data[i] for i in range(tensor.size)]).reshape(tensor.shape)

def make_tensor(data, shape=None, requires_grad=False):
    """Create a Tensor from a list of floats with optional shape and gradient tracking."""
    arr = (c_float * len(data))(*data)
    return Tensor(arr, requires_grad=requires_grad, shape=shape or (len(data),))

make_ctypes = lambda lst: (c_float * len(lst))(*lst)

def numerical_grad(f, x_tensor, idx):
    """Compute numerical gradient for a tensor at a specific index using finite differences."""
    orig = x_tensor.data[idx]
    x_tensor.data[idx] = orig + EPS
    f1 = f().data[0]
    x_tensor.data[idx] = orig - EPS
    f2 = f().data[0]
    x_tensor.data[idx] = orig
    return (f1 - f2) / (2 * EPS)

def grad_check(test, f, x_tensor, name="", f_numpy=None):
    """Check forward pass and gradients against NumPy implementation."""
    x_tensor.requires_grad = True
    y = f()

    if f_numpy:
        x_np = to_numpy(x_tensor)
        y_np = f_numpy(x_np)
        y_val = to_numpy(y).item()
        print(f"{name} forward check: got {y_val}, expected {y_np}")
        test.assertTrue(np.allclose(y_val, y_np, atol=1e-4),
                        f"{name} forward check failed: got {y_val}, expected {y_np}")

    y.backward()
    print(f"Testing gradients for: {name}")
    for i in range(x_tensor.size):
        expected = numerical_grad(f, x_tensor, i)
        actual = x_tensor.grad[i]
        rel_error = abs(actual - expected) / (abs(expected) + 1e-6)
        test.assertTrue(rel_error < TOL,
                        f"Grad check failed at idx {i}: expected {expected:.5f}, got {actual:.5f}, rel_error={rel_error:.5f}")
    print("Passed.\n")

class TestTensor(unittest.TestCase):
    def test_add(self):
        """Test element-wise addition and its gradients."""
        a = make_tensor([1.0, 2.0, 3.0], requires_grad=True)
        b_np = np.array([4.0, 5.0, 6.0])
        b = make_tensor(b_np.tolist())
        def f(): return (a + b).sum()
        grad_check(self, f, a, "add", lambda x: np.sum(x + b_np))

    def test_mul(self):
        """Test element-wise multiplication and its gradients."""
        a = make_tensor([1.0, 2.0, 3.0], requires_grad=True)
        b_np = np.array([0.1, 0.2, 0.3])
        b = make_tensor(b_np.tolist())
        def f(): return (a * b).sum()
        grad_check(self, f, a, "mul", lambda x: np.sum(x * b_np))

    def test_pow(self):
        """Test element-wise power operation and its gradients."""
        a = make_tensor([1.0, 2.0, 3.0], requires_grad=True)
        def f(): return (a ** 2).sum()
        grad_check(self, f, a, "pow", lambda x: np.sum(x ** 2))

    def test_sum(self):
        """Test sum reduction and its gradients."""
        a = make_tensor([1.0, -2.0, 3.0], requires_grad=True)
        def f(): return a.sum()
        grad_check(self, f, a, "sum", lambda x: np.sum(x))

    def test_mean(self):
        """Test mean reduction and its gradients."""
        a = make_tensor([1.0, -2.0, 3.0], requires_grad=True)
        def f(): return a.mean()
        grad_check(self, f, a, "mean", lambda x: np.mean(x))

    def test_matmul(self):
        """Test matrix multiplication and its gradients."""
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_np = np.array([[1.0, 0.0], [0.0, 1.0]])
        a = make_tensor(a_np.flatten().tolist(), requires_grad=True, shape=(2, 2))
        b = make_tensor(b_np.flatten().tolist(), shape=(2, 2))
        def f(): return a.matmul(b).sum()
        grad_check(self, f, a, "matmul", lambda x: np.sum(np.matmul(x.reshape(2, 2), b_np)))

    def test_relu(self):
        """Test ReLU activation and its gradients."""
        a = make_tensor([-1.0, -0.1, 1.0], requires_grad=True)
        def f(): return a.relu().sum()
        grad_check(self, f, a, "relu", lambda x: np.sum(np.maximum(0, x)))

    def test_cross_entropy(self):
        """Test cross-entropy loss and its gradients."""
        logits_np = np.array([[2.0, 1.0, 0.1], [0.3, 2.5, 0.3]])
        targets_np = np.array([0, 2])
        logits = make_tensor(logits_np.flatten().tolist(), requires_grad=True, shape=(2, 3))
        targets = make_tensor(targets_np.tolist(), shape=(2,))
        def f(): return logits.cross_entropy(targets)
        def f_np(x):
            x = x.reshape(2, 3)
            exp = np.exp(x - x.max(axis=1, keepdims=True))
            probs = exp / exp.sum(axis=1, keepdims=True)
            log_likelihood = -np.log(probs[np.arange(2), targets_np])
            return np.mean(log_likelihood)
        grad_check(self, f, logits, "cross_entropy", f_np)

    def test_broadcast_to_shape(self):
        """Test broadcasting to various shapes."""
        # Test 1: Scalar to 1D vector
        data = make_ctypes([1.0])
        result = broadcast_to_shape(data, (1,), (3,), 1)
        self.assertEqual([result[i] for i in range(3)], [1.0, 1.0, 1.0])

        # Test 2: Scalar to 2D matrix
        data = make_ctypes([2.0])
        result = broadcast_to_shape(data, (1,), (2, 3), 1)
        self.assertEqual([result[i] for i in range(6)], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

        # Test 3: 1D vector to 2D matrix (broadcast along rows)
        data = make_ctypes([1.0, 2.0, 3.0])
        result = broadcast_to_shape(data, (3,), (2, 3), 3)
        self.assertEqual([result[i] for i in range(6)], [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])

        # Test 4: 2D matrix to 3D tensor (broadcast along batch dimension)
        data = make_ctypes([1.0, 2.0, 3.0, 4.0])
        result = broadcast_to_shape(data, (2, 2), (3, 2, 2), 4)
        self.assertEqual([result[i] for i in range(12)], [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])

        # Test 5: Invalid broadcast (incompatible shapes)
        data = make_ctypes([1.0, 2.0])
        with self.assertRaises(ValueError):
            broadcast_to_shape(data, (2,), (2, 3), 2)

    def test_add_broadcast(self):
        """Test broadcasting in addition and gradient computation."""
        a = make_tensor(([1.0, 2.0, 3.0]), requires_grad=True, shape=(3,))
        b = make_tensor(([1.0]), shape=(1,))
        c = a + b
        self.assertEqual(c.shape, (3,))
        loss = c.sum()
        loss.backward()
        self.assertEqual([a.grad[i] for i in range(3)], [1.0, 1.0, 1.0])

    def test_reshape_and_grad(self):
        """Test reshape operation and gradient propagation."""
        a = make_tensor(([1.0, 2.0, 3.0, 4.0]), requires_grad=True, shape=(4,))
        b = a.reshape((2, 2))
        c = b.sum()
        c.backward()
        self.assertEqual([a.grad[i] for i in range(4)], [1.0, 1.0, 1.0, 1.0])

    def test_matmul_batch(self):
        """Test batched matrix multiplication and gradients."""
        a = make_tensor(([1, 2, 3, 4, 5, 6]), requires_grad=True, shape=(2, 3))
        b = make_tensor(([1, 1, 1]), shape=(3, 1))
        c = a.matmul(b)
        self.assertEqual(c.shape, (2, 1))
        c.sum().backward()
        self.assertEqual([a.grad[i] for i in range(6)], [1, 1, 1, 1, 1, 1])

    def test_cross_entropy(self):
        """Test cross-entropy loss and gradient computation."""
        logits = make_tensor(([2.0, 1.0, 0.1, 0.5, 2.5, 0.3]), requires_grad=True, shape=(2, 3))
        labels = make_tensor(([0.0, 2.0]), shape=(2,))
        loss = logits.cross_entropy(labels)
        loss.backward()
        self.assertEqual(loss.shape, (1,))
        self.assertIsNotNone(logits.grad)

    def test_serialization_roundtrip(self):
        """Test tensor serialization and deserialization."""
        a = make_tensor(([1.0, 2.0, 3.0]), requires_grad=True, shape=(3,))
        a.grad = make_ctypes([1.0, 1.0, 1.0])
        serialized = pickle.dumps(a)
        b = pickle.loads(serialized)
        self.assertEqual([b.data[i] for i in range(3)], [1.0, 2.0, 3.0])
        self.assertEqual([b.grad[i] for i in range(3)], [1.0, 1.0, 1.0])

    def test_flatten_and_backward(self):
        """Test flatten operation and gradient propagation."""
        a = make_tensor(list(range(6)), requires_grad=True, shape=(2, 3))
        b = a.flatten()
        c = b.sum()
        c.backward()
        self.assertEqual([a.grad[i] for i in range(6)], [1.0] * 6)

    def test_avgpool2d_and_maxpool2d(self):
        """Test 2D average and max pooling with gradient computation."""
        x = make_tensor([float(i) for i in range(16)], requires_grad=True, shape=(1, 1, 4, 4))
        y1 = x.avgpool2d(kernel_size=2)
        y2 = x.maxpool2d(kernel_size=2)
        self.assertEqual(y1.shape, (1, 1, 2, 2))
        self.assertEqual(y2.shape, (1, 1, 2, 2))
        (y1 + y2).sum().backward()
        self.assertIsNotNone(x.grad)

if __name__ == "__main__":
    unittest.main()