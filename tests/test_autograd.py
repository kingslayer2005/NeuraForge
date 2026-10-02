"""
test_autograd.py — Comprehensive tests for the autograd engine.

Tests are organized into three sections required by the Phase 1 gate:
    (a) Numerical gradient check on every op (central difference, float64, rtol<1e-6)
    (b) Diamond dependency test (y = x*x + x, dy/dx == 2x + 1)
    (c) Broadcasting test suite for every binary op across all shape combinations

Additional tests:
    - Topological ordering correctness
    - Softmax / log_softmax numerical stability
    - Fused softmax-cross-entropy backward
    - Reduction ops (sum, mean, max)
    - Shape ops (reshape, transpose, getitem, concatenate)
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuraforge.autograd import (
    Tensor, concatenate, stack, softmax_cross_entropy, mse_loss,
    no_grad, unbroadcast
)
from neuraforge.seed import seed_everything


# ---------------------------------------------------------------------------
# Helper: Numerical gradient check via central differences
# ---------------------------------------------------------------------------
def numerical_gradient_check(
    func,
    inputs,
    eps=1e-5,
    rtol=1e-6,
    atol=1e-8,
):
    """Check analytical gradients against central finite differences.

    Parameters
    ----------
    func : callable
        Takes a list of np.ndarray and returns a scalar float.
    inputs : list of np.ndarray
        The input arrays. Must be float64 for numerical precision.

    Returns
    -------
    list of np.ndarray
        The numerical gradients, one per input.
    """
    numerical_grads = []
    for input_idx, x in enumerate(inputs):
        grad = np.zeros_like(x)
        x_flat = x.reshape(-1)
        grad_flat = grad.reshape(-1)

        for i in range(len(x_flat)):
            orig = x_flat[i]

            x_flat[i] = orig + eps
            loss_plus = func(inputs)

            x_flat[i] = orig - eps
            loss_minus = func(inputs)

            x_flat[i] = orig
            grad_flat[i] = (loss_plus - loss_minus) / (2.0 * eps)

        numerical_grads.append(grad.reshape(x.shape))
    return numerical_grads


def check_op_gradient(func_tensor, func_numpy, inputs_np, names=None):
    """End-to-end gradient check for a tensor operation.

    Parameters
    ----------
    func_tensor : callable
        Takes a list of Tensors, returns a scalar Tensor.
    func_numpy : callable
        Takes a list of np.ndarrays, returns a scalar float.
    inputs_np : list of np.ndarray
        Input arrays (float64).
    """
    # 1. Compute analytical gradients via autograd
    tensors = [Tensor(x.copy(), requires_grad=True) for x in inputs_np]
    result = func_tensor(tensors)
    result.backward()
    analytical_grads = [t.grad for t in tensors]

    # 2. Compute numerical gradients via central differences
    numerical_grads = numerical_gradient_check(func_numpy, [x.copy() for x in inputs_np])

    # 3. Compare
    for i, (ag, ng) in enumerate(zip(analytical_grads, numerical_grads)):
        name = names[i] if names else f"input_{i}"
        np.testing.assert_allclose(
            ag, ng, rtol=1e-6, atol=1e-8,
            err_msg=f"Gradient mismatch for {name}"
        )


@pytest.fixture(autouse=True)
def set_seed():
    seed_everything(42)


# ===================================================================
# (a) NUMERICAL GRADIENT CHECK ON EVERY OP
# ===================================================================

class TestUnaryOpGradients:
    """Test gradient of every unary operation via finite differences."""

    def test_exp(self):
        x = np.random.randn(3, 4).astype(np.float64)
        check_op_gradient(
            lambda ts: ts[0].exp().sum(),
            lambda xs: np.sum(np.exp(xs[0])),
            [x]
        )

    def test_log(self):
        # Use positive values to avoid log(negative)
        x = np.abs(np.random.randn(3, 4).astype(np.float64)) + 0.1
        check_op_gradient(
            lambda ts: ts[0].log().sum(),
            lambda xs: np.sum(np.log(xs[0])),
            [x]
        )

    def test_sqrt(self):
        x = np.abs(np.random.randn(3, 4).astype(np.float64)) + 0.1
        check_op_gradient(
            lambda ts: ts[0].sqrt().sum(),
            lambda xs: np.sum(np.sqrt(xs[0])),
            [x]
        )

    def test_abs(self):
        # Avoid values near 0 where abs is not differentiable
        x = np.random.randn(3, 4).astype(np.float64)
        x[np.abs(x) < 0.1] = 0.5
        check_op_gradient(
            lambda ts: ts[0].abs().sum(),
            lambda xs: np.sum(np.abs(xs[0])),
            [x]
        )

    def test_tanh(self):
        x = np.random.randn(3, 4).astype(np.float64)
        check_op_gradient(
            lambda ts: ts[0].tanh().sum(),
            lambda xs: np.sum(np.tanh(xs[0])),
            [x]
        )

    def test_sigmoid(self):
        x = np.random.randn(3, 4).astype(np.float64)
        check_op_gradient(
            lambda ts: ts[0].sigmoid().sum(),
            lambda xs: np.sum(1.0 / (1.0 + np.exp(-xs[0]))),
            [x]
        )

    def test_relu(self):
        x = np.random.randn(3, 4).astype(np.float64)
        x[np.abs(x) < 0.1] = 0.5  # Avoid non-differentiable point at 0
        check_op_gradient(
            lambda ts: ts[0].relu().sum(),
            lambda xs: np.sum(np.maximum(0, xs[0])),
            [x]
        )

    def test_gelu(self):
        x = np.random.randn(3, 4).astype(np.float64)
        import math
        SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
        COEFF = 0.044715

        def numpy_gelu(xs):
            z = xs[0]
            inner = SQRT_2_OVER_PI * (z + COEFF * z ** 3)
            return np.sum(0.5 * z * (1.0 + np.tanh(inner)))

        check_op_gradient(
            lambda ts: ts[0].gelu().sum(),
            numpy_gelu,
            [x]
        )

    def test_silu(self):
        x = np.random.randn(3, 4).astype(np.float64)
        check_op_gradient(
            lambda ts: ts[0].silu().sum(),
            lambda xs: np.sum(xs[0] / (1.0 + np.exp(-xs[0]))),
            [x]
        )

    def test_neg(self):
        x = np.random.randn(3, 4).astype(np.float64)
        check_op_gradient(
            lambda ts: (-ts[0]).sum(),
            lambda xs: np.sum(-xs[0]),
            [x]
        )

    def test_power(self):
        x = np.abs(np.random.randn(3, 4).astype(np.float64)) + 0.1
        check_op_gradient(
            lambda ts: (ts[0] ** 2.5).sum(),
            lambda xs: np.sum(xs[0] ** 2.5),
            [x]
        )


class TestBinaryOpGradients:
    """Test gradient of every binary operation via finite differences."""

    def test_add(self):
        a = np.random.randn(3, 4).astype(np.float64)
        b = np.random.randn(3, 4).astype(np.float64)
        check_op_gradient(
            lambda ts: (ts[0] + ts[1]).sum(),
            lambda xs: np.sum(xs[0] + xs[1]),
            [a, b]
        )

    def test_sub(self):
        a = np.random.randn(3, 4).astype(np.float64)
        b = np.random.randn(3, 4).astype(np.float64)
        check_op_gradient(
            lambda ts: (ts[0] - ts[1]).sum(),
            lambda xs: np.sum(xs[0] - xs[1]),
            [a, b]
        )

    def test_mul(self):
        a = np.random.randn(3, 4).astype(np.float64)
        b = np.random.randn(3, 4).astype(np.float64)
        check_op_gradient(
            lambda ts: (ts[0] * ts[1]).sum(),
            lambda xs: np.sum(xs[0] * xs[1]),
            [a, b]
        )

    def test_div(self):
        a = np.random.randn(3, 4).astype(np.float64)
        b = np.abs(np.random.randn(3, 4).astype(np.float64)) + 0.5  # avoid div by zero
        check_op_gradient(
            lambda ts: (ts[0] / ts[1]).sum(),
            lambda xs: np.sum(xs[0] / xs[1]),
            [a, b]
        )

    def test_matmul(self):
        a = np.random.randn(3, 5).astype(np.float64)
        b = np.random.randn(5, 4).astype(np.float64)
        check_op_gradient(
            lambda ts: (ts[0] @ ts[1]).sum(),
            lambda xs: np.sum(xs[0] @ xs[1]),
            [a, b]
        )


class TestReductionGradients:
    """Test gradient of sum, mean, max."""

    def test_sum_all(self):
        x = np.random.randn(3, 4).astype(np.float64)
        check_op_gradient(
            lambda ts: ts[0].sum(),
            lambda xs: np.sum(xs[0]),
            [x]
        )

    def test_sum_axis0(self):
        x = np.random.randn(3, 4).astype(np.float64)
        check_op_gradient(
            lambda ts: ts[0].sum(axis=0).sum(),
            lambda xs: np.sum(np.sum(xs[0], axis=0)),
            [x]
        )

    def test_sum_axis1_keepdims(self):
        x = np.random.randn(3, 4).astype(np.float64)
        check_op_gradient(
            lambda ts: ts[0].sum(axis=1, keepdims=True).sum(),
            lambda xs: np.sum(np.sum(xs[0], axis=1, keepdims=True)),
            [x]
        )

    def test_mean_all(self):
        x = np.random.randn(3, 4).astype(np.float64)
        check_op_gradient(
            lambda ts: ts[0].mean(),
            lambda xs: np.mean(xs[0]),
            [x]
        )

    def test_mean_axis0(self):
        x = np.random.randn(3, 4).astype(np.float64)
        check_op_gradient(
            lambda ts: ts[0].mean(axis=0).sum(),
            lambda xs: np.sum(np.mean(xs[0], axis=0)),
            [x]
        )

    def test_max_all(self):
        # Use distinct values to avoid ties at non-differentiable points
        x = np.array([1.0, 5.0, 3.0, 2.0, 4.0, 6.0]).reshape(2, 3).astype(np.float64)
        check_op_gradient(
            lambda ts: ts[0].max(),
            lambda xs: np.max(xs[0]),
            [x]
        )

    def test_max_axis1(self):
        x = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]]).astype(np.float64)
        check_op_gradient(
            lambda ts: ts[0].max(axis=1).sum(),
            lambda xs: np.sum(np.max(xs[0], axis=1)),
            [x]
        )


class TestShapeOpGradients:
    """Test gradient of reshape, transpose, getitem, concatenate."""

    def test_reshape(self):
        x = np.random.randn(3, 4).astype(np.float64)
        check_op_gradient(
            lambda ts: ts[0].reshape(12).sum(),
            lambda xs: np.sum(xs[0].reshape(12)),
            [x]
        )

    def test_transpose(self):
        x = np.random.randn(3, 4).astype(np.float64)
        check_op_gradient(
            lambda ts: ts[0].transpose().sum(),
            lambda xs: np.sum(xs[0].T),
            [x]
        )

    def test_getitem_slice(self):
        x = np.random.randn(5, 4).astype(np.float64)
        check_op_gradient(
            lambda ts: ts[0][1:3].sum(),
            lambda xs: np.sum(xs[0][1:3]),
            [x]
        )

    def test_getitem_index(self):
        x = np.random.randn(5, 4).astype(np.float64)
        check_op_gradient(
            lambda ts: ts[0][2].sum(),
            lambda xs: np.sum(xs[0][2]),
            [x]
        )

    def test_concatenate(self):
        a = np.random.randn(3, 4).astype(np.float64)
        b = np.random.randn(2, 4).astype(np.float64)
        check_op_gradient(
            lambda ts: concatenate([ts[0], ts[1]], axis=0).sum(),
            lambda xs: np.sum(np.concatenate([xs[0], xs[1]], axis=0)),
            [a, b]
        )


class TestSoftmaxLogSoftmaxGradients:
    """Test softmax and log_softmax gradients."""

    def test_softmax(self):
        x = np.random.randn(3, 5).astype(np.float64)

        def np_softmax(xs):
            shifted = xs[0] - np.max(xs[0], axis=1, keepdims=True)
            e = np.exp(shifted)
            return np.sum(e / np.sum(e, axis=1, keepdims=True))

        check_op_gradient(
            lambda ts: ts[0].softmax(axis=1).sum(),
            np_softmax,
            [x]
        )

    def test_log_softmax(self):
        x = np.random.randn(3, 5).astype(np.float64)

        def np_log_softmax(xs):
            x_max = np.max(xs[0], axis=1, keepdims=True)
            shifted = xs[0] - x_max
            lse = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
            return np.sum(shifted - lse)

        check_op_gradient(
            lambda ts: ts[0].log_softmax(axis=1).sum(),
            np_log_softmax,
            [x]
        )

    def test_fused_softmax_cross_entropy(self):
        logits_np = np.random.randn(4, 5).astype(np.float64)
        targets = np.eye(5)[np.array([0, 1, 3, 2])].astype(np.float64)

        def np_softmax_ce(xs):
            shifted = xs[0] - np.max(xs[0], axis=1, keepdims=True)
            exp_s = np.exp(shifted)
            probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
            return -np.sum(targets * np.log(probs + 1e-12)) / xs[0].shape[0]

        check_op_gradient(
            lambda ts: softmax_cross_entropy(ts[0], targets),
            np_softmax_ce,
            [logits_np]
        )


# ===================================================================
# (b) DIAMOND DEPENDENCY TEST
# ===================================================================

class TestDiamondDependencies:
    """Test that gradients accumulate correctly when one tensor feeds
    into multiple downstream operations."""

    def test_diamond_y_equals_x_times_x_plus_x(self):
        """y = x * x + x  →  dy/dx = 2x + 1"""
        for val in [0.0, 1.0, -2.0, 3.5, -0.7]:
            x = Tensor(np.array([val]), requires_grad=True)
            y = x * x + x
            y.backward()

            expected = 2 * val + 1
            np.testing.assert_allclose(
                x.grad, np.array([expected]), rtol=1e-10,
                err_msg=f"Diamond test failed for x={val}"
            )

    def test_diamond_multiple_uses(self):
        """z = x + x + x  →  dz/dx = 3"""
        x = Tensor(np.array([5.0]), requires_grad=True)
        z = x + x + x
        z.backward()
        np.testing.assert_allclose(x.grad, np.array([3.0]), rtol=1e-10)

    def test_diamond_complex(self):
        """y = x^2 + 2*x + 1  →  dy/dx = 2*x + 2"""
        x_val = 3.0
        x = Tensor(np.array([x_val]), requires_grad=True)
        y = x ** 2 + x * 2.0 + 1.0
        y.backward()
        expected = 2 * x_val + 2
        np.testing.assert_allclose(x.grad, np.array([expected]), rtol=1e-10)

    def test_diamond_vector(self):
        """y = sum(x * x + x) where x is a vector."""
        x_val = np.array([1.0, 2.0, 3.0])
        x = Tensor(x_val, requires_grad=True)
        y = (x * x + x).sum()
        y.backward()
        # dy/dx_i = 2*x_i + 1
        expected = 2 * x_val + 1
        np.testing.assert_allclose(x.grad, expected, rtol=1e-10)


# ===================================================================
# (c) BROADCASTING TEST SUITE
# ===================================================================

class TestBroadcastingGradients:
    """For each binary op, check every combination of shapes:
    (1,), (n,), (1,m), (n,1), (n,m)
    """

    # Shape combinations to test
    SHAPE_PAIRS = [
        # (shape_a, shape_b) — all broadcasting combos
        ((1,),   (4,)),       # scalar-like broadcast
        ((4,),   (1,)),       # reverse
        ((4,),   (4,)),       # no broadcast needed
        ((1, 4), (3, 4)),     # (1,m) vs (n,m)
        ((3, 4), (1, 4)),     # (n,m) vs (1,m)
        ((3, 1), (3, 4)),     # (n,1) vs (n,m)
        ((3, 4), (3, 1)),     # (n,m) vs (n,1)
        ((1, 1), (3, 4)),     # scalar-like vs full
        ((3, 4), (1, 1)),     # full vs scalar-like
        ((4,),   (3, 4)),     # (m,) vs (n,m) — leading dim added
        ((3, 4), (4,)),       # (n,m) vs (m,)
    ]

    @pytest.mark.parametrize("shape_a,shape_b", SHAPE_PAIRS)
    def test_add_broadcast(self, shape_a, shape_b):
        a = np.random.randn(*shape_a).astype(np.float64)
        b = np.random.randn(*shape_b).astype(np.float64)
        check_op_gradient(
            lambda ts: (ts[0] + ts[1]).sum(),
            lambda xs: np.sum(xs[0] + xs[1]),
            [a, b]
        )

    @pytest.mark.parametrize("shape_a,shape_b", SHAPE_PAIRS)
    def test_sub_broadcast(self, shape_a, shape_b):
        a = np.random.randn(*shape_a).astype(np.float64)
        b = np.random.randn(*shape_b).astype(np.float64)
        check_op_gradient(
            lambda ts: (ts[0] - ts[1]).sum(),
            lambda xs: np.sum(xs[0] - xs[1]),
            [a, b]
        )

    @pytest.mark.parametrize("shape_a,shape_b", SHAPE_PAIRS)
    def test_mul_broadcast(self, shape_a, shape_b):
        a = np.random.randn(*shape_a).astype(np.float64)
        b = np.random.randn(*shape_b).astype(np.float64)
        check_op_gradient(
            lambda ts: (ts[0] * ts[1]).sum(),
            lambda xs: np.sum(xs[0] * xs[1]),
            [a, b]
        )

    @pytest.mark.parametrize("shape_a,shape_b", SHAPE_PAIRS)
    def test_div_broadcast(self, shape_a, shape_b):
        a = np.random.randn(*shape_a).astype(np.float64)
        b = np.abs(np.random.randn(*shape_b).astype(np.float64)) + 0.5
        check_op_gradient(
            lambda ts: (ts[0] / ts[1]).sum(),
            lambda xs: np.sum(xs[0] / xs[1]),
            [a, b]
        )


# ===================================================================
# ADDITIONAL: Numerical stability, no_grad, etc.
# ===================================================================

class TestNumericalStability:
    """Verify softmax and log_softmax don't overflow on extreme inputs."""

    def test_softmax_large_inputs(self):
        """Softmax on inputs in the thousands must not overflow."""
        x = Tensor(np.array([[1000.0, 2000.0, 3000.0]]), requires_grad=True)
        p = x.softmax(axis=1)
        # The result should be valid probabilities (no inf or nan)
        assert np.all(np.isfinite(p.data)), f"Softmax produced non-finite values: {p.data}"
        # Last element should be ~1.0 since it's largest by far
        np.testing.assert_allclose(p.data[0, 2], 1.0, atol=1e-10)

    def test_log_softmax_large_inputs(self):
        """log_softmax on inputs in the thousands must not produce -inf."""
        x = Tensor(np.array([[1000.0, 2000.0, 3000.0]]), requires_grad=True)
        lp = x.log_softmax(axis=1)
        assert np.all(np.isfinite(lp.data)), f"Log-softmax produced non-finite values: {lp.data}"

    def test_softmax_negative_large_inputs(self):
        """Softmax on very negative inputs must not underflow to NaN."""
        x = Tensor(np.array([[-1000.0, -2000.0, -3000.0]]), requires_grad=True)
        p = x.softmax(axis=1)
        assert np.all(np.isfinite(p.data))
        # First element should be ~1.0
        np.testing.assert_allclose(p.data[0, 0], 1.0, atol=1e-10)


class TestNoGrad:
    """Test the no_grad() context manager."""

    def test_no_grad_disables_graph(self):
        x = Tensor(np.array([3.0]), requires_grad=True)
        with no_grad():
            y = x * x
        # y should not have a backward function
        assert y._backward_fn is None

    def test_grad_re_enabled_after_context(self):
        x = Tensor(np.array([3.0]), requires_grad=True)
        with no_grad():
            _ = x * x
        # After context, graph tracking should resume
        y = x * x
        assert y._backward_fn is not None


class TestUnbroadcastHelper:
    """Direct tests for the unbroadcast helper function."""

    def test_no_broadcast(self):
        g = np.ones((3, 4))
        result = unbroadcast(g, (3, 4))
        assert result.shape == (3, 4)

    def test_leading_dim(self):
        g = np.ones((3, 4))
        result = unbroadcast(g, (4,))
        assert result.shape == (4,)
        np.testing.assert_allclose(result, np.full(4, 3.0))

    def test_size_1_axis(self):
        g = np.ones((3, 4))
        result = unbroadcast(g, (3, 1))
        assert result.shape == (3, 1)
        np.testing.assert_allclose(result, np.full((3, 1), 4.0))

    def test_scalar_target(self):
        g = np.ones((3, 4))
        result = unbroadcast(g, (1, 1))
        assert result.shape == (1, 1)
        np.testing.assert_allclose(result, np.array([[12.0]]))


class TestCompositeForwardBackward:
    """Test that composite chains produce correct gradients."""

    def test_linear_layer_manual(self):
        """Simulate a Dense layer: y = X @ W + b, loss = sum(y)."""
        X = np.random.randn(4, 5).astype(np.float64)
        W = np.random.randn(5, 3).astype(np.float64)
        b = np.random.randn(1, 3).astype(np.float64)

        X_t = Tensor(X)
        W_t = Tensor(W, requires_grad=True)
        b_t = Tensor(b, requires_grad=True)

        y = X_t @ W_t + b_t
        loss = y.sum()
        loss.backward()

        # Analytical gradients for y = X @ W + b, loss = sum(y)
        # dL/dW = X^T @ ones, dL/db = sum(ones, axis=0)
        expected_dW = X.T @ np.ones((4, 3))
        expected_db = np.sum(np.ones((4, 3)), axis=0, keepdims=True)

        np.testing.assert_allclose(W_t.grad, expected_dW, rtol=1e-10)
        np.testing.assert_allclose(b_t.grad, expected_db, rtol=1e-10)

    def test_mlp_gradient_check(self):
        """Full MLP: Dense -> ReLU -> Dense -> sum. Check all gradients."""
        np.random.seed(42)
        X = np.random.randn(4, 5).astype(np.float64)
        W1 = np.random.randn(5, 8).astype(np.float64)
        b1 = np.zeros((1, 8)).astype(np.float64)
        W2 = np.random.randn(8, 3).astype(np.float64)
        b2 = np.zeros((1, 3)).astype(np.float64)

        def forward_np(params):
            w1, b1_p, w2, b2_p = params
            h = X @ w1 + b1_p
            h = np.maximum(0, h)  # ReLU
            out = h @ w2 + b2_p
            return np.sum(out)

        def forward_tensor(params):
            w1_t, b1_t, w2_t, b2_t = params
            X_t = Tensor(X)
            h = (X_t @ w1_t + b1_t).relu()
            out = h @ w2_t + b2_t
            return out.sum()

        check_op_gradient(
            forward_tensor,
            forward_np,
            [W1, b1, W2, b2],
            names=["W1", "b1", "W2", "b2"]
        )
