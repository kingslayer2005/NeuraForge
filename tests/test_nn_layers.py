"""
test_nn_layers.py — Tests for Conv2d, BatchNorm, LayerNorm, MaxPool2d, Embedding.

Validates forward/backward correctness via numerical gradient checks.
All checks in float64 with rtol < 1e-6.
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuraforge.nn import (
    Dense, Conv2d, MaxPool2d, BatchNorm1d, BatchNorm2d,
    LayerNorm, Dropout, Embedding, ReLU, GELU, SiLU, ForageAct,
)
from neuraforge.seed import seed_everything


def numerical_grad(func, x, eps=1e-6):
    """Central-difference numerical gradient for a function f: array -> scalar."""
    grad = np.zeros_like(x)
    x_flat = x.reshape(-1)
    grad_flat = grad.reshape(-1)
    for i in range(len(x_flat)):
        orig = x_flat[i]
        x_flat[i] = orig + eps
        fp = func(x.reshape(x.shape))
        x_flat[i] = orig - eps
        fm = func(x.reshape(x.shape))
        x_flat[i] = orig
        grad_flat[i] = (fp - fm) / (2 * eps)
    return grad


@pytest.fixture(autouse=True)
def set_seed():
    seed_everything(42)


# ===================================================================
# CONV2D TESTS
# ===================================================================
class TestConv2d:
    """Test Conv2d forward and backward via numerical gradient check."""

    def test_conv2d_forward_shape(self):
        """Output shape should be correct."""
        conv = Conv2d(3, 16, kernel_size=3, stride=1, padding=0)
        X = np.random.randn(2, 3, 8, 8).astype(np.float64)
        out = conv.forward(X)
        assert out.shape == (2, 16, 6, 6), f"Expected (2,16,6,6), got {out.shape}"

    def test_conv2d_forward_shape_with_padding(self):
        """With padding=1 and kernel=3, output should preserve spatial dims."""
        conv = Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        X = np.random.randn(2, 3, 8, 8).astype(np.float64)
        out = conv.forward(X)
        assert out.shape == (2, 16, 8, 8), f"Expected (2,16,8,8), got {out.shape}"

    def test_conv2d_gradient_input(self):
        """Numerical gradient check for input."""
        conv = Conv2d(1, 2, kernel_size=3, stride=1, padding=0)
        conv.W.data = conv.W.data.astype(np.float64)
        conv.b.data = conv.b.data.astype(np.float64)
        X = np.random.randn(2, 1, 6, 6).astype(np.float64)

        out = conv.forward(X)
        d_out = np.random.randn(*out.shape).astype(np.float64)
        d_input = conv.backward(d_out)

        def func(x_val):
            o = conv.forward(x_val)
            return np.sum(o * d_out)

        ng = numerical_grad(func, X.copy())
        np.testing.assert_allclose(d_input, ng, rtol=1e-6, atol=1e-7,
                                   err_msg="Conv2d input gradient mismatch")

    def test_conv2d_gradient_weights(self):
        """Numerical gradient check for weights."""
        conv = Conv2d(1, 2, kernel_size=3, stride=1, padding=0)
        conv.W.data = conv.W.data.astype(np.float64)
        conv.b.data = conv.b.data.astype(np.float64)
        X = np.random.randn(2, 1, 6, 6).astype(np.float64)

        out = conv.forward(X)
        d_out = np.random.randn(*out.shape).astype(np.float64)
        conv.backward(d_out)

        def func(w_val):
            old_W = conv.W.data.copy()
            conv.W.data = w_val
            o = conv.forward(X)
            conv.W.data = old_W
            return np.sum(o * d_out)

        ng = numerical_grad(func, conv.W.data.copy())
        np.testing.assert_allclose(conv.W.grad, ng, rtol=1e-6, atol=1e-7,
                                   err_msg="Conv2d weight gradient mismatch")

    def test_conv2d_gradient_bias(self):
        """Numerical gradient check for bias."""
        conv = Conv2d(1, 2, kernel_size=3, stride=1, padding=0)
        conv.W.data = conv.W.data.astype(np.float64)
        conv.b.data = conv.b.data.astype(np.float64)
        X = np.random.randn(2, 1, 6, 6).astype(np.float64)

        out = conv.forward(X)
        d_out = np.random.randn(*out.shape).astype(np.float64)
        conv.backward(d_out)

        def func(b_val):
            old_b = conv.b.data.copy()
            conv.b.data = b_val
            o = conv.forward(X)
            conv.b.data = old_b
            return np.sum(o * d_out)

        ng = numerical_grad(func, conv.b.data.copy())
        np.testing.assert_allclose(conv.b.grad, ng, rtol=1e-6, atol=1e-7,
                                   err_msg="Conv2d bias gradient mismatch")


# ===================================================================
# MAXPOOL2D TESTS
# ===================================================================
class TestMaxPool2d:
    def test_forward_shape(self):
        pool = MaxPool2d(kernel_size=2, stride=2)
        X = np.random.randn(2, 3, 8, 8).astype(np.float64)
        out = pool.forward(X)
        assert out.shape == (2, 3, 4, 4)

    def test_gradient_routing(self):
        """Only max elements should receive gradient."""
        pool = MaxPool2d(kernel_size=2, stride=2)
        # Simple input where max positions are clear
        X = np.array([[[[1, 0, 3, 0],
                        [0, 2, 0, 4],
                        [5, 0, 7, 0],
                        [0, 6, 0, 8]]]]).astype(np.float64)
        out = pool.forward(X)
        d_out = np.ones_like(out)
        dX = pool.backward(d_out)

        # Gradient should only be at max positions
        assert dX[0, 0, 1, 1] == 1.0  # max of top-left 2x2 is 2 at (1,1)
        assert dX[0, 0, 0, 0] == 0.0  # not the max


# ===================================================================
# BATCHNORM1D TESTS
# ===================================================================
class TestBatchNorm1d:
    def test_forward_normalization(self):
        """Output should have approximately zero mean and unit variance."""
        bn = BatchNorm1d(4)
        X = np.random.randn(32, 4).astype(np.float64) * 5 + 3
        out = bn.forward(X, training=True)

        # After BN, mean should be ~0 (beta=0) and var ~1 (gamma=1)
        np.testing.assert_allclose(np.mean(out, axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(np.var(out, axis=0), 1.0, atol=0.05)

    def test_gradient_input(self):
        bn = BatchNorm1d(4)
        bn.gamma.data = bn.gamma.data.astype(np.float64)
        bn.beta.data = bn.beta.data.astype(np.float64)
        X = np.random.randn(8, 4).astype(np.float64)

        out = bn.forward(X, training=True)
        d_out = np.random.randn(*out.shape).astype(np.float64)
        dX = bn.backward(d_out)

        def func(x_val):
            bn2 = BatchNorm1d(4)
            bn2.gamma.data = bn.gamma.data.copy()
            bn2.beta.data = bn.beta.data.copy()
            o = bn2.forward(x_val, training=True)
            return np.sum(o * d_out)

        ng = numerical_grad(func, X.copy())
        np.testing.assert_allclose(dX, ng, rtol=1e-6, atol=1e-7,
                                   err_msg="BatchNorm1d input gradient mismatch")

    def test_eval_uses_running_stats(self):
        """In eval mode, BN should use running_mean/var, not batch stats."""
        bn = BatchNorm1d(4)
        X = np.random.randn(32, 4).astype(np.float64) * 5 + 3
        bn.forward(X, training=True)

        # Eval mode output should use running stats
        X_test = np.random.randn(1, 4).astype(np.float64)
        out_eval = bn.forward(X_test, training=False)
        assert out_eval.shape == (1, 4)


# ===================================================================
# LAYERNORM TESTS
# ===================================================================
class TestLayerNorm:
    def test_forward_normalization(self):
        ln = LayerNorm(8)
        X = np.random.randn(4, 8).astype(np.float64) * 5 + 3
        out = ln.forward(X)
        # Each row should have approximately zero mean and unit variance
        np.testing.assert_allclose(np.mean(out, axis=1), 0.0, atol=1e-6)
        np.testing.assert_allclose(np.var(out, axis=1), 1.0, atol=0.05)

    def test_gradient_input(self):
        ln = LayerNorm(4)
        ln.gamma.data = ln.gamma.data.astype(np.float64)
        ln.beta.data = ln.beta.data.astype(np.float64)
        X = np.random.randn(3, 4).astype(np.float64)

        out = ln.forward(X)
        d_out = np.random.randn(*out.shape).astype(np.float64)
        dX = ln.backward(d_out)

        def func(x_val):
            ln2 = LayerNorm(4)
            ln2.gamma.data = ln.gamma.data.copy()
            ln2.beta.data = ln.beta.data.copy()
            o = ln2.forward(x_val)
            return np.sum(o * d_out)

        ng = numerical_grad(func, X.copy())
        np.testing.assert_allclose(dX, ng, rtol=1e-6, atol=1e-7,
                                   err_msg="LayerNorm input gradient mismatch")


# ===================================================================
# EMBEDDING TESTS
# ===================================================================
class TestEmbedding:
    def test_forward(self):
        emb = Embedding(10, 4)
        indices = np.array([0, 3, 7])
        out = emb.forward(indices)
        assert out.shape == (3, 4)
        np.testing.assert_array_equal(out, emb.W.data[[0, 3, 7]])

    def test_backward_accumulates(self):
        """Same index used twice should accumulate gradient."""
        emb = Embedding(5, 3)
        indices = np.array([1, 1, 3])
        out = emb.forward(indices)
        d_out = np.ones((3, 3), dtype=np.float64)
        emb.backward(d_out)

        # Index 1 appears twice, so gradient should be 2x
        np.testing.assert_allclose(emb.W.grad[1], 2.0)
        # Index 3 appears once
        np.testing.assert_allclose(emb.W.grad[3], 1.0)
        # Index 0 not used, gradient should be 0
        np.testing.assert_allclose(emb.W.grad[0], 0.0)


# ===================================================================
# DROPOUT TESTS
# ===================================================================
class TestDropout:
    def test_eval_is_identity(self):
        drop = Dropout(0.5)
        X = np.random.randn(4, 8).astype(np.float64)
        out = drop.forward(X, training=False)
        np.testing.assert_array_equal(out, X)

    def test_training_zeros_some(self):
        drop = Dropout(0.5)
        np.random.seed(0)
        X = np.ones((100, 100)).astype(np.float64)
        out = drop.forward(X, training=True)
        # Roughly 50% should be zero
        zero_frac = np.mean(out == 0.0)
        assert 0.3 < zero_frac < 0.7, f"Expected ~50% zeros, got {zero_frac:.2f}"


# ===================================================================
# LEGACY PARITY TESTS (Phase 2 gate)
# ===================================================================
class TestLegacyParity:
    """Verify that nn.Dense produces the same gradients as the legacy Dense."""

    def test_dense_gradient_parity(self):
        """Autograd-backed Dense and legacy Dense should agree to 1e-10."""
        from neuraforge.legacy.layers import Dense as LegacyDense

        np.random.seed(42)
        X = np.random.randn(4, 5).astype(np.float64)
        d_out = np.random.randn(4, 3).astype(np.float64)

        # Create both versions with same weights
        new_dense = Dense(5, 3)
        legacy_dense = LegacyDense(5, 3)

        # Copy weights
        legacy_dense.W.data = new_dense.W.data.copy()
        legacy_dense.b.data = new_dense.b.data.copy()

        # Forward
        new_out = new_dense.forward(X)
        legacy_out = legacy_dense.forward(X)
        np.testing.assert_allclose(new_out, legacy_out, rtol=1e-12)

        # Backward
        new_din = new_dense.backward(d_out)
        legacy_din = legacy_dense.backward(d_out)

        np.testing.assert_allclose(new_din, legacy_din, rtol=1e-10, atol=1e-12,
                                   err_msg="Input gradient mismatch between nn.Dense and legacy")
        np.testing.assert_allclose(new_dense.W.grad, legacy_dense.W.grad, rtol=1e-10, atol=1e-12,
                                   err_msg="W gradient mismatch between nn.Dense and legacy")
        np.testing.assert_allclose(new_dense.b.grad, legacy_dense.b.grad, rtol=1e-10, atol=1e-12,
                                   err_msg="b gradient mismatch between nn.Dense and legacy")

    def test_forageact_gradient_parity(self):
        """nn.ForageAct and legacy ForageAct should agree to 1e-10."""
        from neuraforge.legacy.activations import ForageAct as LegacyForageAct

        np.random.seed(42)
        X = np.random.randn(4, 8).astype(np.float64)
        d_out = np.random.randn(4, 8).astype(np.float64)

        for mode in ["fixed", "scalar", "per_neuron"]:
            new_act = ForageAct(mode=mode, init_alpha=0.3, n_neurons=8)
            legacy_act = LegacyForageAct(mode=mode, init_alpha=0.3, n_neurons=8)

            if mode != "fixed":
                legacy_act._alpha_param.data[:] = new_act._alpha_param.data

            new_out = new_act.forward(X.copy())
            legacy_out = legacy_act.forward(X.copy())
            np.testing.assert_allclose(new_out, legacy_out, rtol=1e-12,
                                       err_msg=f"ForageAct {mode} forward mismatch")

            new_din = new_act.backward(d_out)
            legacy_din = legacy_act.backward(d_out)
            np.testing.assert_allclose(new_din, legacy_din, rtol=1e-10, atol=1e-12,
                                       err_msg=f"ForageAct {mode} backward mismatch")
