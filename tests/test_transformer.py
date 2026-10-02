"""
test_transformer.py — Tests for transformer components.

Validates:
    1. Multi-head attention produces correct shapes
    2. Causal masking prevents attending to future positions
    3. Full transformer block forward/backward does not crash or produce NaN
    4. Gradient check on a small transformer block via numerical differences
    5. DecoderTransformer end-to-end forward/backward
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from neuraforge.seed import seed_everything
from neuraforge.transformer import (
    DecoderTransformer,
    FeedForward,
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    TransformerBlock,
    scaled_dot_product_attention,
)


@pytest.fixture(autouse=True)
def set_seed():
    seed_everything(42)


class TestScaledDotProductAttention:

    def test_output_shapes(self):
        batch, n_heads, seq, d_k = 2, 4, 8, 16
        Q = np.random.randn(batch, n_heads, seq, d_k).astype(np.float64)
        K = np.random.randn(batch, n_heads, seq, d_k).astype(np.float64)
        V = np.random.randn(batch, n_heads, seq, d_k).astype(np.float64)

        out, weights = scaled_dot_product_attention(Q, K, V)
        assert out.shape == (batch, n_heads, seq, d_k)
        assert weights.shape == (batch, n_heads, seq, seq)

    def test_weights_sum_to_one(self):
        """Attention weights should sum to 1 along the key dimension."""
        Q = np.random.randn(2, 1, 5, 8).astype(np.float64)
        K = np.random.randn(2, 1, 5, 8).astype(np.float64)
        V = np.random.randn(2, 1, 5, 8).astype(np.float64)

        _, weights = scaled_dot_product_attention(Q, K, V)
        sums = np.sum(weights, axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-10)

    def test_causal_mask(self):
        """With causal mask, attention to future positions should be ~0."""
        seq = 4
        Q = np.random.randn(1, 1, seq, 8).astype(np.float64)
        K = np.random.randn(1, 1, seq, 8).astype(np.float64)
        V = np.random.randn(1, 1, seq, 8).astype(np.float64)

        # Causal mask: upper triangular = True (masked)
        mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
        mask = mask[np.newaxis, np.newaxis, :, :]

        _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

        # Check: weights for future positions should be ~0
        for i in range(seq):
            for j in range(i + 1, seq):
                assert weights[0, 0, i, j] < 1e-6, \
                    f"Position {i} attends to future position {j}: weight={weights[0,0,i,j]}"


class TestMultiHeadAttention:

    def test_output_shape(self):
        mha = MultiHeadAttention(d_model=64, n_heads=4)
        x = np.random.randn(2, 8, 64).astype(np.float64)
        out = mha.forward(x)
        assert out.shape == (2, 8, 64)

    def test_backward_shapes(self):
        """Backward should produce gradients of correct shapes."""
        mha = MultiHeadAttention(d_model=32, n_heads=4)
        x = np.random.randn(2, 6, 32).astype(np.float64)
        out = mha.forward(x)
        d_out = np.random.randn(*out.shape).astype(np.float64)
        d_input = mha.backward(d_out)
        assert d_input.shape == x.shape

    def test_all_params_get_gradients(self):
        """Every parameter should receive a gradient after backward."""
        mha = MultiHeadAttention(d_model=32, n_heads=4)
        x = np.random.randn(2, 6, 32).astype(np.float64)
        out = mha.forward(x)
        d_out = np.ones_like(out)
        mha.backward(d_out)

        for p in mha.parameters():
            assert p.grad is not None, f"Parameter {p.name} has no gradient"
            assert not np.all(p.grad == 0), f"Parameter {p.name} has all-zero gradient"


class TestSinusoidalPE:

    def test_shape(self):
        pe = SinusoidalPositionalEncoding(d_model=64)
        x = np.zeros((2, 10, 64))
        out = pe.forward(x)
        assert out.shape == (2, 10, 64)

    def test_different_positions_different_encoding(self):
        pe = SinusoidalPositionalEncoding(d_model=32)
        x = np.zeros((1, 5, 32))
        out = pe.forward(x)
        # Different positions should have different encodings
        for i in range(5):
            for j in range(i + 1, 5):
                assert not np.allclose(out[0, i], out[0, j]), \
                    f"Positions {i} and {j} have identical encodings"


class TestFeedForward:

    def test_output_shape(self):
        ffn = FeedForward(d_model=32, d_ff=128)
        x = np.random.randn(2, 8, 32).astype(np.float64)
        out = ffn.forward(x)
        assert out.shape == (2, 8, 32)

    def test_backward_shapes(self):
        ffn = FeedForward(d_model=32, d_ff=128)
        x = np.random.randn(2, 8, 32).astype(np.float64)
        out = ffn.forward(x)
        d_out = np.random.randn(*out.shape).astype(np.float64)
        d_input = ffn.backward(d_out)
        assert d_input.shape == x.shape


class TestTransformerBlock:

    def test_output_shape(self):
        block = TransformerBlock(d_model=64, n_heads=4, d_ff=256)
        x = np.random.randn(2, 8, 64).astype(np.float64)
        out = block.forward(x)
        assert out.shape == (2, 8, 64)

    def test_no_nan(self):
        block = TransformerBlock(d_model=64, n_heads=4, d_ff=256)
        x = np.random.randn(2, 8, 64).astype(np.float64)
        out = block.forward(x)
        assert not np.any(np.isnan(out)), "Transformer block produced NaN"

    def test_backward_no_crash(self):
        """Full forward/backward through a transformer block."""
        block = TransformerBlock(d_model=32, n_heads=4, d_ff=128)
        x = np.random.randn(2, 6, 32).astype(np.float64)
        out = block.forward(x)
        d_out = np.random.randn(*out.shape).astype(np.float64)
        d_input = block.backward(d_out)
        assert d_input.shape == x.shape
        assert not np.any(np.isnan(d_input))

    def test_all_params_get_gradients(self):
        block = TransformerBlock(d_model=32, n_heads=4, d_ff=128)
        x = np.random.randn(2, 6, 32).astype(np.float64)
        out = block.forward(x)
        d_out = np.ones_like(out)
        block.backward(d_out)

        for p in block.parameters():
            assert p.grad is not None, f"Parameter {p.name} has no gradient"


class TestDecoderTransformer:

    def test_forward_shape(self):
        model = DecoderTransformer(
            vocab_size=50, d_model=32, n_heads=4,
            n_layers=2, d_ff=128, max_seq_len=16, dropout=0.0
        )
        tokens = np.random.randint(0, 50, (2, 10))
        logits = model.forward(tokens)
        assert logits.shape == (2, 10, 50)

    def test_backward_no_crash(self):
        model = DecoderTransformer(
            vocab_size=50, d_model=32, n_heads=4,
            n_layers=2, d_ff=128, max_seq_len=16, dropout=0.0
        )
        tokens = np.random.randint(0, 50, (2, 10))
        logits = model.forward(tokens, training=True)
        d_logits = np.random.randn(*logits.shape).astype(np.float64)
        model.backward(d_logits)

        # All parameters should have gradients
        for p in model.parameters():
            assert p.grad is not None, f"Parameter {p.name} has no gradient"

    def test_training_step_reduces_loss(self):
        """A single training step should reduce the loss."""
        from neuraforge.optimizers import Adam

        model = DecoderTransformer(
            vocab_size=26, d_model=32, n_heads=4,
            n_layers=2, d_ff=64, max_seq_len=16, dropout=0.0
        )
        params = model.parameters()
        opt = Adam(params, lr=1e-3)

        # Dummy data: predict next token
        tokens = np.random.randint(0, 26, (4, 8))
        targets = np.eye(26)[tokens[:, 1:]]  # shift by 1

        # Forward
        logits = model.forward(tokens[:, :-1], training=True)

        # Compute loss (softmax cross-entropy)
        B, T, V = logits.shape
        logits_2d = logits.reshape(-1, V)
        targets_2d = targets.reshape(-1, 26)

        shifted = logits_2d - np.max(logits_2d, axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        loss_before = -np.sum(targets_2d * np.log(probs + 1e-12)) / (B * T)

        # Backward
        d_logits = ((probs - targets_2d) / (B * T)).reshape(B, T, V)
        opt.zero_grad()
        model.backward(d_logits)
        opt.step()

        # Forward again
        logits2 = model.forward(tokens[:, :-1], training=False)
        logits2_2d = logits2.reshape(-1, V)
        shifted2 = logits2_2d - np.max(logits2_2d, axis=1, keepdims=True)
        exp_s2 = np.exp(shifted2)
        probs2 = exp_s2 / np.sum(exp_s2, axis=1, keepdims=True)
        loss_after = -np.sum(targets_2d * np.log(probs2 + 1e-12)) / (B * T)

        assert loss_after < loss_before, f"Loss did not decrease: {loss_before:.4f} → {loss_after:.4f}"

    def test_generate(self):
        model = DecoderTransformer(
            vocab_size=26, d_model=32, n_heads=4,
            n_layers=2, d_ff=64, max_seq_len=16, dropout=0.0
        )
        start = np.array([[0, 1, 2]])
        generated = model.generate(start, max_new_tokens=5, temperature=1.0)
        assert generated.shape == (1, 8)  # 3 start + 5 new
        assert np.all(generated >= 0) and np.all(generated < 26)
