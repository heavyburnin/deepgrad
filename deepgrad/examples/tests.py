import unittest
import math
import random
import numpy as np
from typing import Tuple
from deepgrad.tensor import Tensor, zeros, ones, rand, randn

class TestDeepGradTensor(unittest.TestCase):
    def assertTensorEqual(self, tensor: Tensor, expected: np.ndarray, tol: float = 1e-6):
        """Helper to compare tensor data with expected values within tolerance."""
        actual = np.array(tensor.data).reshape(tensor.shape)
        np.testing.assert_array_almost_equal(actual, expected, decimal=int(-math.log10(tol)))
        self.assertEqual(tensor.shape, expected.shape)

    def assertGradEqual(self, tensor: Tensor, expected_grad: np.ndarray, tol: float = 1e-6):
        """Helper to compare tensor gradients with expected values within tolerance."""
        if tensor.grad is None:
            self.assertIsNone(expected_grad)
            return
        actual = np.array(tensor.grad).reshape(tensor.shape)
        np.testing.assert_array_almost_equal(actual, expected_grad, decimal=int(-math.log10(tol)))
        self.assertEqual(tensor.shape, expected_grad.shape)

    def setUp(self):
        """Set up test fixtures."""
        random.seed(42)  # For reproducible random tests
        np.random.seed(42)

    def test_tensor_creation(self):
        """Test tensor creation with various inputs."""
        # Basic creation
        data = [1.0, 2.0, 3.0]
        t = Tensor(data)
        self.assertEqual(t.shape, (3,))
        self.assertEqual(t.size, 3)
        self.assertFalse(t.requires_grad)
        self.assertTensorEqual(t, np.array(data))

        # With shape
        data_2d = [1.0, 2.0, 3.0, 4.0]
        t = Tensor(data_2d, shape=(2, 2))
        self.assertEqual(t.shape, (2, 2))
        self.assertEqual(t.size, 4)
        self.assertTensorEqual(t, np.array(data_2d).reshape(2, 2))

        # Invalid shape
        with self.assertRaises(ValueError):
            Tensor([1.0, 2.0], shape=(3,))

        # Invalid data type
        with self.assertRaises(ValueError):
            Tensor("invalid")

    def test_zeros_ones(self):
        """Test zeros and ones creation functions."""
        shape = (2, 3)
        t1 = zeros(shape)
        self.assertEqual(t1.shape, shape)
        self.assertTensorEqual(t1, np.zeros(shape))

        shape = (2, 2)
        t2 = ones(shape)
        self.assertEqual(t2.shape, shape)
        self.assertTensorEqual(t2, np.ones(shape))

    def test_rand_randn(self):
        """Test random tensor creation."""
        shape = (2, 2)
        t1 = rand(shape)
        self.assertEqual(t1.shape, shape)
        self.assertTrue(np.all(np.array(t1.data).reshape(shape) >= 0.0))
        self.assertTrue(np.all(np.array(t1.data).reshape(shape) < 1.0))

        shape = (1000,)
        t2 = randn(shape, mean=0.0, std=1.0)
        data = np.array(t2.data).reshape(shape)
        self.assertAlmostEqual(np.mean(data), 0.0, delta=0.1)
        self.assertAlmostEqual(np.std(data), 1.0, delta=0.1)

    def test_arithmetic_operations(self):
        """Test basic arithmetic operations and their gradients."""
        a_data = np.array([1.0, 2.0, 3.0])
        b_data = np.array([4.0, 5.0, 6.0])
        a = Tensor(a_data.tolist(), requires_grad=True)
        b = Tensor(b_data.tolist(), requires_grad=True)

        # Addition
        c = a + b
        self.assertTensorEqual(c, a_data + b_data)
        c.sum().backward()
        self.assertGradEqual(a, np.ones_like(a_data))
        self.assertGradEqual(b, np.ones_like(b_data))

        # Subtraction
        a.grad, b.grad = None, None
        c = a - b
        self.assertTensorEqual(c, a_data - b_data)
        c.sum().backward()
        self.assertGradEqual(a, np.ones_like(a_data))
        self.assertGradEqual(b, -np.ones_like(b_data))

        # Multiplication
        a.grad, b.grad = None, None
        c = a * b
        self.assertTensorEqual(c, a_data * b_data)
        c.sum().backward()
        self.assertGradEqual(a, b_data)
        self.assertGradEqual(b, a_data)

        # Division
        a.grad, b.grad = None, None
        c = a / b
        self.assertTensorEqual(c, a_data / b_data)
        c.sum().backward()
        self.assertGradEqual(a, 1.0 / b_data, tol=1e-5)
        self.assertGradEqual(b, -a_data / (b_data ** 2), tol=1e-5)

        # Power
        a.grad, b.grad = None, None
        c = a ** 2
        self.assertTensorEqual(c, a_data ** 2)
        c.sum().backward()
        self.assertGradEqual(a, 2.0 * a_data)

    def test_broadcasting(self):
        """Test broadcasting and gradient unbroadcasting."""
        a_data = np.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2)
        b_data = np.array([2.0, 3.0])
        a = Tensor(a_data.flatten().tolist(), shape=(2, 2), requires_grad=True)
        b = Tensor(b_data.tolist(), requires_grad=True)
        c = a + b
        self.assertEqual(c.shape, (2, 2))
        self.assertTensorEqual(c, a_data + b_data)
        c.sum().backward()
        self.assertGradEqual(a, np.ones_like(a_data))
        self.assertGradEqual(b, np.sum(np.ones_like(a_data), axis=0))

    def test_reshape(self):
        """Test reshape operation and gradient propagation."""
        a_data = np.array([1.0, 2.0, 3.0, 4.0])
        a = Tensor(a_data.tolist(), requires_grad=True)
        b = a.reshape((2, 2))
        self.assertEqual(b.shape, (2, 2))
        self.assertTensorEqual(b, a_data.reshape(2, 2))
        b.sum().backward()
        self.assertGradEqual(a, np.ones_like(a_data))

        # Test invalid reshape
        with self.assertRaises(ValueError):
            a.reshape((3, 2))

    def test_flatten(self):
        """Test flatten operation."""
        a_data = np.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2)
        a = Tensor(a_data.flatten().tolist(), shape=(2, 2), requires_grad=True)
        b = a.flatten()
        self.assertEqual(b.shape, (4,))
        self.assertTensorEqual(b, a_data.flatten())
        b.sum().backward()
        self.assertGradEqual(a, np.ones_like(a_data))

    def test_matmul(self):
        """Test matrix multiplication and gradients."""
        # Basic matmul
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        b_data = np.array([[5.0, 6.0], [7.0, 8.0]])
        a = Tensor(a_data.flatten().tolist(), shape=(2, 2), requires_grad=True)
        b = Tensor(b_data.flatten().tolist(), shape=(2, 2), requires_grad=True)

        c = a.matmul(b)
        self.assertEqual(c.shape, (2, 2))
        self.assertTensorEqual(c, a_data @ b_data)
        c.sum().backward()
        self.assertGradEqual(a, np.ones_like(c.data).reshape(c.shape) @ b_data.T)
        self.assertGradEqual(b, a_data.T @ np.ones_like(c.data).reshape(c.shape))

        # Batched matmul
        a_data = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])  # shape: (2, 1, 2)
        b_data = np.array([[[5.0], [6.0]], [[7.0], [8.0]]])  # shape: (2, 2, 1)

        a = Tensor(a_data.flatten().tolist(), shape=(2, 1, 2), requires_grad=True)
        b = Tensor(b_data.copy().reshape(-1).tolist(), shape=(2, 2, 1), requires_grad=True)
        c = a.matmul(b)  # shape: (2, 1, 1)

        self.assertEqual(c.shape, (2, 1, 1))
        expected = np.matmul(a_data, b_data)
        self.assertTensorEqual(c, expected)
        c.sum().backward()

        grad_out = np.ones_like(c.data).reshape(2, 1, 1)
        expected_grad_a = np.matmul(grad_out, b_data.transpose(0, 2, 1))
        expected_grad_b = np.matmul(a_data.transpose(0, 2, 1), grad_out)
        self.assertGradEqual(a, expected_grad_a)
        self.assertGradEqual(b, expected_grad_b)

    def test_conv2d(self):
        """Test 2D convolution and gradients."""
        input_data = np.array([1.0, 2.0, 3.0, 4.0]).reshape(1, 1, 2, 2)
        weight_data = np.array([1.0, 0.0, 0.0, 1.0]).reshape(1, 1, 2, 2)
        bias_data = np.array([0.0])
        input = Tensor(input_data.flatten().tolist(), shape=(1, 1, 2, 2), requires_grad=True)
        weight = Tensor(weight_data.flatten().tolist(), shape=(1, 1, 2, 2), requires_grad=True)
        bias = Tensor(bias_data.tolist(), requires_grad=True)
        out = input.conv2d(weight, bias, stride=(1, 1), padding=(0, 0))
        self.assertEqual(out.shape, (1, 1, 1, 1))
        expected_out = np.sum(input_data * weight_data) + bias_data
        self.assertTensorEqual(out, expected_out.reshape(1, 1, 1, 1))
        out.sum().backward()
        self.assertGradEqual(input, weight_data)
        self.assertGradEqual(weight, input_data)
        self.assertGradEqual(bias, np.ones_like(bias_data))

    def test_maxpool2d(self):
        """Test 2D max pooling and gradients."""
        input_data = np.array([1.0, 2.0, 3.0, 4.0]).reshape(1, 1, 2, 2)
        input = Tensor(input_data.flatten().tolist(), shape=(1, 1, 2, 2), requires_grad=True)
        out = input.maxpool2d(kernel_size=(2, 2))
        self.assertEqual(out.shape, (1, 1, 1, 1))
        expected_out = np.max(input_data)
        self.assertTensorEqual(out, np.array([expected_out]).reshape(1, 1, 1, 1))
        out.sum().backward()
        expected_grad = np.zeros_like(input_data)
        expected_grad[np.unravel_index(np.argmax(input_data), input_data.shape)] = 1.0
        self.assertGradEqual(input, expected_grad)

    def test_avgpool2d(self):
        """Test 2D average pooling and gradients."""
        input_data = np.array([1.0, 2.0, 3.0, 4.0]).reshape(1, 1, 2, 2)
        input = Tensor(input_data.flatten().tolist(), shape=(1, 1, 2, 2), requires_grad=True)
        out = input.avgpool2d(kernel_size=(2, 2))
        self.assertEqual(out.shape, (1, 1, 1, 1))
        expected_out = np.mean(input_data)
        self.assertTensorEqual(out, np.array([expected_out]).reshape(1, 1, 1, 1))
        out.sum().backward()
        expected_grad = np.ones_like(input_data) / input_data.size
        self.assertGradEqual(input, expected_grad)

    def test_relu(self):
        """Test ReLU activation and gradients."""
        a_data = np.array([-1.0, 0.0, 1.0, 2.0])
        a = Tensor(a_data.tolist(), requires_grad=True)
        b = a.relu()
        self.assertTensorEqual(b, np.maximum(a_data, 0.0))
        b.sum().backward()
        expected_grad = np.where(a_data > 0, 1.0, 0.0)
        self.assertGradEqual(a, expected_grad)

    def test_cross_entropy(self):
        """Test cross-entropy loss and gradients."""
        logits_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(2, 3)
        target_data = np.array([1, 2])
        logits = Tensor(logits_data.flatten().tolist(), shape=(2, 3), requires_grad=True)
        target = Tensor(target_data.tolist(), requires_grad=False)
        loss = logits.cross_entropy(target)
        
        # Compute expected loss
        exps = np.exp(logits_data - np.max(logits_data, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        expected_loss = -np.mean(np.log(probs[np.arange(2), target_data]))
        self.assertAlmostEqual(loss.data[0], expected_loss, delta=1e-4)
        
        loss.backward()
        # Compute expected gradient: (softmax(logits) - one_hot(targets)) / batch_size
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(2), target_data] = 1.0
        expected_grad = (probs - one_hot) / 2
        self.assertGradEqual(logits, expected_grad, tol=1e-4)

    def test_dropout(self):
        """Test dropout operation."""
        random.seed(42)
        np.random.seed(42)
        a_data = np.array([1.0, 2.0, 3.0, 4.0])
        a = Tensor(a_data.tolist(), requires_grad=True)
        p = 0.5
        random.seed(42)  # Ensure Python's random matches test setup
        b = a.dropout(p=p)
        # Use the mask stored in the output tensor
        mask = np.array(b._mask).reshape(a_data.shape) if b._mask is not None else np.ones_like(a_data)
        scale = 1.0 / (1.0 - p)
        expected = a_data * mask
        self.assertTensorEqual(b, expected, tol=1e-6)
        b.sum().backward()
        self.assertGradEqual(a, mask, tol=1e-6)

    def test_clone_detach(self):
        """Test clone and detach operations."""
        a_data = np.array([1.0, 2.0])
        a = Tensor(a_data.tolist(), requires_grad=True)
        b = a.clone()
        self.assertTensorEqual(b, a_data)
        self.assertTrue(b.requires_grad)
        b.sum().backward()
        self.assertIsNone(a.grad)

        c = a.detach()
        self.assertTensorEqual(c, a_data)
        self.assertFalse(c.requires_grad)

    def test_backward_non_scalar(self):
        """Test error handling for non-scalar backward."""
        a = Tensor([1.0, 2.0], requires_grad=True)
        with self.assertRaises(RuntimeError):
            a.backward()

    def test_release_graph(self):
        """Test releasing computation graph memory."""
        a_data = np.array([1.0, 2.0])
        b_data = np.array([3.0, 4.0])
        a = Tensor(a_data.tolist(), requires_grad=True)
        b = Tensor(b_data.tolist(), requires_grad=True)
        c = a * b
        loss = c.sum()
        loss.backward()
        c.release_graph()
        self.assertIsNone(c._backward)
        self.assertEqual(c._prev, [])
        self.assertIsNone(a.grad)
        self.assertIsNone(b.grad)

if __name__ == '__main__':
    unittest.main()