"""
transformer.py — Transformer components built on the NeuraForge nn module.

Implements:
    - Scaled dot-product attention with optional causal mask
    - Multi-head attention with correct head splitting and merging
    - Positional encoding (sinusoidal and learned)
    - Transformer block: attention + residual + LayerNorm + FFN + residual + LayerNorm
    - Full decoder-only transformer model with tied input/output embeddings

All components use hand-written forward/backward methods operating on NumPy
arrays, following the same protocol as the rest of NeuraForge.
"""

from __future__ import annotations

import math

import numpy as np

from neuraforge.layers import Parameter
from neuraforge.nn import Dropout, Embedding, LayerNorm, Module


# ===================================================================
# SCALED DOT-PRODUCT ATTENTION
# ===================================================================
def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Parameters
    ----------
    Q : np.ndarray, shape (..., seq_q, d_k)
    K : np.ndarray, shape (..., seq_k, d_k)
    V : np.ndarray, shape (..., seq_k, d_v)
    mask : np.ndarray, optional
        Boolean mask where True means "mask out" (set to -inf before softmax).
        Shape: (..., seq_q, seq_k) or broadcastable.

    Returns
    -------
    output : np.ndarray, shape (..., seq_q, d_v)
    attn_weights : np.ndarray, shape (..., seq_q, seq_k)
        The softmax attention weights (useful for visualization).
    """
    d_k = Q.shape[-1]

    # Step 1: Compute attention scores
    # scores shape: (..., seq_q, seq_k)
    scores = Q @ K.swapaxes(-2, -1) / math.sqrt(d_k)

    # Step 2: Apply causal mask (set masked positions to -inf)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    # Step 3: Softmax over the key dimension (last axis)
    # Subtract max for numerical stability
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Step 4: Weighted sum of values
    # output shape: (..., seq_q, d_v)
    output = attn_weights @ V

    return output, attn_weights


# ===================================================================
# MULTI-HEAD ATTENTION
# ===================================================================
class MultiHeadAttention(Module):
    """Multi-head attention mechanism.

    Splits Q, K, V into multiple heads, applies scaled dot-product attention
    independently on each head, then concatenates and projects the result.

    Head splitting:
        Input (batch, seq, d_model) → reshape to (batch, seq, n_heads, d_k)
        → transpose to (batch, n_heads, seq, d_k)
        This gives each head its own independent attention computation.

    Parameters
    ----------
    d_model : int
        Model dimension (input and output).
    n_heads : int
        Number of attention heads.
    dropout : float
        Attention dropout probability. Default 0.0.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension per head
        self.dropout_rate = dropout

        # Projection matrices for Q, K, V, and output
        # W_q, W_k, W_v: (d_model, d_model) — projects input to Q, K, V
        # W_o: (d_model, d_model) — projects concatenated heads back to d_model
        scale = np.sqrt(2.0 / d_model)
        self.W_q = Parameter("W_q", np.random.randn(d_model, d_model) * scale)
        self.b_q = Parameter("b_q", np.zeros(d_model))
        self.W_k = Parameter("W_k", np.random.randn(d_model, d_model) * scale)
        self.b_k = Parameter("b_k", np.zeros(d_model))
        self.W_v = Parameter("W_v", np.random.randn(d_model, d_model) * scale)
        self.b_v = Parameter("b_v", np.zeros(d_model))
        self.W_o = Parameter("W_o", np.random.randn(d_model, d_model) * scale)
        self.b_o = Parameter("b_o", np.zeros(d_model))

    @property
    def config(self) -> dict:
        return {"type": "MultiHeadAttention", "d_model": self.d_model,
                "n_heads": self.n_heads}

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """Split the last dimension into (n_heads, d_k).

        Input:  (batch, seq, d_model)
        Output: (batch, n_heads, seq, d_k)
        """
        batch, seq, _ = x.shape
        x = x.reshape(batch, seq, self.n_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """Merge heads back into a single dimension.

        Input:  (batch, n_heads, seq, d_k)
        Output: (batch, seq, d_model)
        """
        batch, _, seq, _ = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch, seq, self.d_model)

    def forward(self, x: np.ndarray, training: bool = True,
                mask: np.ndarray | None = None) -> np.ndarray:
        """Forward pass for multi-head attention.

        For self-attention: Q = K = V = x (projected through different matrices).

        Parameters
        ----------
        x : np.ndarray, shape (batch, seq, d_model)
        mask : np.ndarray, optional
            Causal or padding mask.

        Returns
        -------
        np.ndarray, shape (batch, seq, d_model)
        """
        _batch, _seq, _ = x.shape
        self._input = x

        # Step 1: Linear projections — Q, K, V
        # Q = x @ W_q + b_q, shape: (batch, seq, d_model)
        Q = x @ self.W_q.data + self.b_q.data
        K = x @ self.W_k.data + self.b_k.data
        V = x @ self.W_v.data + self.b_v.data

        self._Q_full = Q
        self._K_full = K
        self._V_full = V

        # Step 2: Split into heads
        # Each becomes (batch, n_heads, seq, d_k)
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # Step 3: Scaled dot-product attention per head
        attn_out, self._attn_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        # attn_out shape: (batch, n_heads, seq, d_k)

        self._Q_heads = Q
        self._K_heads = K
        self._V_heads = V
        self._attn_out = attn_out

        # Step 4: Merge heads — concatenate back to (batch, seq, d_model)
        merged = self._merge_heads(attn_out)
        self._merged = merged

        # Step 5: Output projection
        output = merged @ self.W_o.data + self.b_o.data

        return output

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Backward pass for multi-head attention.

        Parameters
        ----------
        d_out : np.ndarray, shape (batch, seq, d_model)

        Returns
        -------
        np.ndarray, shape (batch, seq, d_model)
        """
        batch, seq, _ = d_out.shape

        # Gradient through output projection: out = merged @ W_o + b_o
        merged_2d = self._merged.reshape(-1, self.d_model)
        d_out_2d = d_out.reshape(-1, self.d_model)
        # dW_o = merged^T @ d_out (treating batch*seq as a single axis)
        self.W_o.grad = merged_2d.T @ d_out_2d
        self.b_o.grad = np.sum(d_out_2d, axis=0)

        # d_merged = d_out @ W_o^T, shape (batch, seq, d_model)
        d_merged = d_out @ self.W_o.data.T

        # Un-merge heads: (batch, seq, d_model) → (batch, n_heads, seq, d_k)
        d_attn_out = d_merged.reshape(batch, seq, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Backward through attention:
        # attn_out = attn_weights @ V
        # d_attn_weights = d_attn_out @ V^T, shape (batch, n_heads, seq, seq)
        # d_V = attn_weights^T @ d_attn_out, shape (batch, n_heads, seq, d_k)
        d_attn_weights = d_attn_out @ self._V_heads.swapaxes(-2, -1)
        d_V = self._attn_weights.swapaxes(-2, -1) @ d_attn_out

        # Backward through softmax:
        # attn_weights = softmax(scores)
        # d_scores = attn_weights * (d_attn_weights - sum(d_attn_weights * attn_weights, axis=-1, keepdims=True))
        w = self._attn_weights
        sum_dw_w = np.sum(d_attn_weights * w, axis=-1, keepdims=True)
        d_scores = w * (d_attn_weights - sum_dw_w)

        # Backward through score scaling:
        # scores = Q @ K^T / sqrt(d_k)
        # d_Q_from_scores = d_scores @ K / sqrt(d_k)
        # d_K_from_scores = d_scores^T @ Q / sqrt(d_k)
        scale = math.sqrt(self.d_k)
        d_Q = d_scores @ self._K_heads / scale
        d_K = d_scores.swapaxes(-2, -1) @ self._Q_heads / scale

        # Merge heads back for Q, K, V gradients: (batch, n_heads, seq, d_k) → (batch, seq, d_model)
        d_Q_full = d_Q.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)
        d_K_full = d_K.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)
        d_V_full = d_V.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)

        # Backward through linear projections: Q = x @ W_q + b_q
        x = self._input
        x_2d = x.reshape(-1, self.d_model)

        # dW_q = x^T @ d_Q
        self.W_q.grad = x_2d.T @ d_Q_full.reshape(-1, self.d_model)
        self.b_q.grad = np.sum(d_Q_full.reshape(-1, self.d_model), axis=0)

        self.W_k.grad = x_2d.T @ d_K_full.reshape(-1, self.d_model)
        self.b_k.grad = np.sum(d_K_full.reshape(-1, self.d_model), axis=0)

        self.W_v.grad = x_2d.T @ d_V_full.reshape(-1, self.d_model)
        self.b_v.grad = np.sum(d_V_full.reshape(-1, self.d_model), axis=0)

        # d_input = d_Q @ W_q^T + d_K @ W_k^T + d_V @ W_v^T
        d_input = (d_Q_full @ self.W_q.data.T +
                   d_K_full @ self.W_k.data.T +
                   d_V_full @ self.W_v.data.T)

        return d_input

    def parameters(self) -> list[Parameter]:
        return [self.W_q, self.b_q, self.W_k, self.b_k,
                self.W_v, self.b_v, self.W_o, self.b_o]


# ===================================================================
# POSITIONAL ENCODING (SINUSOIDAL)
# ===================================================================
class SinusoidalPositionalEncoding(Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Not learnable — computed once and cached.

    Parameters
    ----------
    d_model : int
        Model dimension.
    max_len : int
        Maximum sequence length to precompute.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model

        # Precompute the positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)
        # Compute the division term: 10000^(2i/d_model) = exp(2i * -log(10000) / d_model)
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)  # Even indices: sin
        pe[:, 1::2] = np.cos(position * div_term)  # Odd indices: cos

        self._pe = pe  # shape (max_len, d_model)

    @property
    def config(self) -> dict:
        return {"type": "SinusoidalPositionalEncoding", "d_model": self.d_model}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Add positional encoding to input embeddings.

        Parameters
        ----------
        x : np.ndarray, shape (batch, seq_len, d_model)

        Returns
        -------
        np.ndarray, shape (batch, seq_len, d_model)
        """
        seq_len = x.shape[1]
        return x + self._pe[:seq_len]

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """PE is additive and fixed, so gradient passes through unchanged."""
        return d_out

    def parameters(self) -> list[Parameter]:
        return []


# ===================================================================
# LEARNED POSITIONAL ENCODING
# ===================================================================
class LearnedPositionalEncoding(Module):
    """Learned positional encoding.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    max_len : int
        Maximum sequence length.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        init_pe = np.random.randn(max_len, d_model) * 0.02
        self.pe = Parameter("pe", init_pe)

    @property
    def config(self) -> dict:
        return {"type": "LearnedPositionalEncoding", "d_model": self.d_model,
                "max_len": self.max_len}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        seq_len = x.shape[1]
        self._seq_len = seq_len
        return x + self.pe.data[:seq_len]

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        # Gradient for learned PE: sum over batch
        if self.pe.grad is None:
            self.pe.grad = np.zeros_like(self.pe.data)
        self.pe.grad[:self._seq_len] += np.sum(d_out, axis=0)
        return d_out

    def parameters(self) -> list[Parameter]:
        return [self.pe]


# ===================================================================
# FEED-FORWARD NETWORK
# ===================================================================
class FeedForward(Module):
    """Position-wise feed-forward network: FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2.

    Parameters
    ----------
    d_model : int
        Model dimension.
    d_ff : int
        Hidden dimension (typically 4 * d_model).
    dropout : float
        Dropout probability.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / d_ff)

        self.W1 = Parameter("W1", np.random.randn(d_model, d_ff) * scale1)
        self.b1 = Parameter("b1", np.zeros(d_ff))
        self.W2 = Parameter("W2", np.random.randn(d_ff, d_model) * scale2)
        self.b2 = Parameter("b2", np.zeros(d_model))
        self.dropout = Dropout(dropout) if dropout > 0 else None

    @property
    def config(self) -> dict:
        return {"type": "FeedForward", "d_model": self.d_model, "d_ff": self.d_ff}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward: x → Linear → GELU → Dropout → Linear.

        Parameters
        ----------
        x : np.ndarray, shape (batch, seq, d_model)
        """
        self._input = x

        # Reshape for matmul: (batch*seq, d_model)
        batch, seq, d = x.shape
        x_2d = x.reshape(-1, d)

        # First linear + GELU activation
        h = x_2d @ self.W1.data + self.b1.data  # (batch*seq, d_ff)
        # GELU activation (tanh approximation)
        SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
        COEFF = 0.044715
        inner = SQRT_2_OVER_PI * (h + COEFF * h ** 3)
        self._tanh_inner = np.tanh(inner)
        h_act = 0.5 * h * (1.0 + self._tanh_inner)
        self._h_pre_act = h  # before activation
        self._h_act = h_act  # after activation

        if self.dropout is not None:
            h_act = self.dropout.forward(h_act, training=training)

        # Second linear
        out = h_act @ self.W2.data + self.b2.data  # (batch*seq, d_model)

        self._h_after_dropout = h_act
        return out.reshape(batch, seq, d)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Backward pass for FFN."""
        batch, seq, d = d_out.shape
        d_out_2d = d_out.reshape(-1, d)

        # Backward through second linear: out = h_act @ W2 + b2
        self.W2.grad = self._h_after_dropout.T @ d_out_2d
        self.b2.grad = np.sum(d_out_2d, axis=0)
        d_h_act = d_out_2d @ self.W2.data.T

        # Backward through dropout
        if self.dropout is not None:
            d_h_act = self.dropout.backward(d_h_act)

        # Backward through GELU
        h = self._h_pre_act
        t = self._tanh_inner
        SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
        COEFF = 0.044715
        g_prime = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * h ** 2)
        sech2 = 1.0 - t ** 2
        gelu_grad = 0.5 * (1.0 + t) + 0.5 * h * sech2 * g_prime
        d_h = d_h_act * gelu_grad

        # Backward through first linear: h = x @ W1 + b1
        x_2d = self._input.reshape(-1, self.d_model)
        self.W1.grad = x_2d.T @ d_h
        self.b1.grad = np.sum(d_h, axis=0)
        d_input = d_h @ self.W1.data.T

        return d_input.reshape(batch, seq, self.d_model)

    def parameters(self) -> list[Parameter]:
        params = [self.W1, self.b1, self.W2, self.b2]
        return params


# ===================================================================
# TRANSFORMER BLOCK
# ===================================================================
class TransformerBlock(Module):
    """A single transformer decoder block.

    Architecture:
        x → LayerNorm → MultiHeadAttention → Dropout → + residual
        → LayerNorm → FeedForward → Dropout → + residual

    Uses pre-LayerNorm (GPT-2 style) which is more stable for training
    than post-LayerNorm.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Feed-forward hidden dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.drop1 = Dropout(dropout) if dropout > 0 else None
        self.drop2 = Dropout(dropout) if dropout > 0 else None

    @property
    def config(self) -> dict:
        return {"type": "TransformerBlock"}

    def forward(self, x: np.ndarray, training: bool = True,
                mask: np.ndarray | None = None) -> np.ndarray:
        """Forward pass through one transformer block.

        Parameters
        ----------
        x : np.ndarray, shape (batch, seq, d_model)
        mask : np.ndarray, optional
            Causal mask.
        """
        # --- Self-attention sub-layer with residual ---
        self._residual1 = x

        # Pre-LayerNorm
        normed = self.ln1.forward(x, training=training)

        # Multi-head attention
        attn_out = self.attn.forward(normed, training=training, mask=mask)

        # Dropout
        if self.drop1 is not None:
            attn_out = self.drop1.forward(attn_out, training=training)

        # Residual connection
        x = x + attn_out

        # --- Feed-forward sub-layer with residual ---
        self._residual2 = x

        # Pre-LayerNorm
        normed = self.ln2.forward(x, training=training)

        # Feed-forward
        ffn_out = self.ffn.forward(normed, training=training)

        # Dropout
        if self.drop2 is not None:
            ffn_out = self.drop2.forward(ffn_out, training=training)

        # Residual connection
        x = x + ffn_out

        return x

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Backward pass through one transformer block."""
        # --- FFN sub-layer backward ---
        # Residual: output = residual2 + ffn_out
        d_ffn_out = d_out
        d_residual2 = d_out

        # Dropout backward
        if self.drop2 is not None:
            d_ffn_out = self.drop2.backward(d_ffn_out)

        # FFN backward
        d_normed2 = self.ffn.backward(d_ffn_out)

        # LayerNorm2 backward
        d_x2 = self.ln2.backward(d_normed2)

        # Add residual gradient
        d_x2 = d_x2 + d_residual2

        # --- Attention sub-layer backward ---
        d_attn_out = d_x2
        d_residual1 = d_x2

        # Dropout backward
        if self.drop1 is not None:
            d_attn_out = self.drop1.backward(d_attn_out)

        # Attention backward
        d_normed1 = self.attn.backward(d_attn_out)

        # LayerNorm1 backward
        d_x1 = self.ln1.backward(d_normed1)

        # Add residual gradient
        d_input = d_x1 + d_residual1

        return d_input

    def parameters(self) -> list[Parameter]:
        params = []
        params.extend(self.ln1.parameters())
        params.extend(self.attn.parameters())
        params.extend(self.ln2.parameters())
        params.extend(self.ffn.parameters())
        return params


# ===================================================================
# DECODER-ONLY TRANSFORMER MODEL
# ===================================================================
class DecoderTransformer(Module):
    """A decoder-only transformer for character-level language modeling.

    Architecture:
        Token embedding + Positional encoding
        → N × TransformerBlock
        → LayerNorm
        → Linear output projection (tied with embedding weights)

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads per block.
    n_layers : int
        Number of transformer blocks.
    d_ff : int
        Feed-forward hidden dimension.
    max_seq_len : int
        Maximum sequence length.
    dropout : float
        Dropout probability.
    tied_embeddings : bool
        If True, tie input embedding weights with output projection.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        tied_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.tied_embeddings = tied_embeddings

        # Token embedding
        self.token_emb = Embedding(vocab_size, d_model)

        # Positional encoding (sinusoidal)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len)

        # Dropout after embedding
        self.emb_dropout = Dropout(dropout) if dropout > 0 else None

        # Transformer blocks
        self.blocks = []
        for _ in range(n_layers):
            self.blocks.append(TransformerBlock(d_model, n_heads, d_ff, dropout))

        # Final layer norm
        self.final_ln = LayerNorm(d_model)

        # Output projection (if not tied with embeddings)
        if not tied_embeddings:
            self.output_proj = Parameter("output_proj",
                                        np.random.randn(d_model, vocab_size) * 0.02)
        else:
            self.output_proj = None  # Will use token_emb.W.data.T

    @property
    def config(self) -> dict:
        return {
            "type": "DecoderTransformer",
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
        }

    def _make_causal_mask(self, seq_len: int) -> np.ndarray:
        """Create a causal (autoregressive) mask.

        Returns a boolean array where True = masked (cannot attend).
        Shape: (1, 1, seq_len, seq_len) for broadcasting over batch and heads.
        """
        # Upper triangular = future positions = True = masked
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        return mask[np.newaxis, np.newaxis, :, :]

    def forward(self, token_ids: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass.

        Parameters
        ----------
        token_ids : np.ndarray of int, shape (batch, seq_len)
            Token indices.

        Returns
        -------
        np.ndarray, shape (batch, seq_len, vocab_size)
            Logits for each position.
        """
        _batch, _seq_len = token_ids.shape

        # Token embedding: look up (batch, seq_len, d_model)
        x = self.token_emb.forward(token_ids, training=training)

        # Scale embeddings by sqrt(d_model) (following Vaswani et al.)
        x = x * math.sqrt(self.d_model)
        self._emb_scale = math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_enc.forward(x, training=training)

        # Embedding dropout
        if self.emb_dropout is not None:
            x = self.emb_dropout.forward(x, training=training)

        self._after_emb = x

        # Causal mask for autoregressive attention
        causal_mask = self._make_causal_mask(_seq_len)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x, training=training, mask=causal_mask)

        # Final layer norm
        x = self.final_ln.forward(x, training=training)
        self._final_normed = x

        # Output projection: (batch, seq_len, d_model) → (batch, seq_len, vocab_size)
        if self.tied_embeddings:
            # Use embedding weight matrix transposed: W_emb^T
            logits = x @ self.token_emb.W.data.T
        else:
            logits = x @ self.output_proj.data

        return logits

    def backward(self, d_logits: np.ndarray) -> None:
        """Backward pass through the entire transformer.

        Parameters
        ----------
        d_logits : np.ndarray, shape (batch, seq_len, vocab_size)
        """
        # Backward through output projection
        x = self._final_normed
        if self.tied_embeddings:
            # logits = x @ W_emb^T
            # d_x = d_logits @ W_emb
            # d_W_emb from output = x^T @ d_logits (accumulated)
            d_x = d_logits @ self.token_emb.W.data
            x_2d = x.reshape(-1, self.d_model)
            d_logits_2d = d_logits.reshape(-1, self.vocab_size)
            # Gradient for the tied weights (will be added to embedding gradient)
            self._d_W_emb_output = x_2d.T @ d_logits_2d  # (d_model, vocab_size) → (vocab_size, d_model).T
        else:
            d_x = d_logits @ self.output_proj.data.T
            x_2d = x.reshape(-1, self.d_model)
            d_logits_2d = d_logits.reshape(-1, self.vocab_size)
            self.output_proj.grad = x_2d.T @ d_logits_2d

        # Backward through final LayerNorm
        d_x = self.final_ln.backward(d_x)

        # Backward through transformer blocks (in reverse order)
        for block in reversed(self.blocks):
            d_x = block.backward(d_x)

        # Backward through embedding dropout
        if self.emb_dropout is not None:
            d_x = self.emb_dropout.backward(d_x)

        # Backward through positional encoding (passthrough)
        d_x = self.pos_enc.backward(d_x)

        # Backward through embedding scale
        d_x = d_x * self._emb_scale

        # Backward through token embedding
        self.token_emb.backward(d_x)

        # Add the output projection gradient to the embedding gradient (tied weights)
        if self.tied_embeddings:
            # d_W_emb_output has shape (d_model, vocab_size), but W has shape (vocab_size, d_model)
            self.token_emb.W.grad += self._d_W_emb_output.T

    def parameters(self) -> list[Parameter]:
        params = []
        params.extend(self.token_emb.parameters())
        if isinstance(self.pos_enc, LearnedPositionalEncoding):
            params.extend(self.pos_enc.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.final_ln.parameters())
        if self.output_proj is not None:
            params.append(self.output_proj)
        return params

    def generate(self, start_tokens: np.ndarray, max_new_tokens: int = 100,
                 temperature: float = 1.0) -> np.ndarray:
        """Generate text autoregressively.

        Parameters
        ----------
        start_tokens : np.ndarray, shape (1, seq_len)
            Starting token IDs.
        max_new_tokens : int
            Number of new tokens to generate.
        temperature : float
            Sampling temperature. Lower = more deterministic.

        Returns
        -------
        np.ndarray, shape (1, seq_len + max_new_tokens)
            The full sequence including generated tokens.
        """
        tokens = start_tokens.copy()

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            context = tokens[:, -self.max_seq_len:]

            # Forward pass (no training)
            logits = self.forward(context, training=False)

            # Take logits for the last position
            next_logits = logits[:, -1, :]  # (1, vocab_size)

            # Apply temperature
            next_logits = next_logits / temperature

            # Softmax to get probabilities
            shifted = next_logits - np.max(next_logits, axis=-1, keepdims=True)
            exp_logits = np.exp(shifted)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            # Sample from the distribution
            next_token = np.array([[np.random.choice(self.vocab_size, p=probs[0])]])

            # Append to sequence
            tokens = np.concatenate([tokens, next_token], axis=1)

        return tokens
