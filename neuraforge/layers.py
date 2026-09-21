"""
layers.py — Neural network layer primitives (Dense, Dropout).

Every layer follows the same protocol:
    - forward(X, training=True)  → output array
    - backward(d_out)            → gradient w.r.t. the layer's input
    - parameters()               → list of Parameter objects owned by this layer

The Parameter class is the fundamental unit of the parameter registry.
Optimizers update param.data in place; backward() sets param.grad.
"""

from __future__ import annotations

from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Parameter — the fundamental trainable-value wrapper
# ---------------------------------------------------------------------------
class Parameter:
    """Wraps a single learnable NumPy array with its gradient.

    Attributes
    ----------
    name : str
        Human-readable name (e.g. "W", "b", "alpha"). The Sequential model
        will prefix this with the layer index to create a globally unique key.
    data : np.ndarray
        The current value of the parameter (mutated in place by optimizers).
    grad : np.ndarray | None
        The gradient computed during the most recent backward pass.
        Initialised to None; set by backward().
    """

    def __init__(self, name: str, data: np.ndarray) -> None:
        self.name: str = name
        # Always store as ndarray so in-place updates work uniformly
        self.data: np.ndarray = np.asarray(data, dtype=data.dtype)
        # Gradient starts as None; backward() will populate it
        self.grad: np.ndarray | None = None

    def __repr__(self) -> str:
        return f"Parameter(name={self.name!r}, shape={self.data.shape})"


# ---------------------------------------------------------------------------
# Dense (fully-connected) layer
# ---------------------------------------------------------------------------
class Dense:
    """Fully-connected linear layer: output = X @ W + b.

    Uses He initialisation (Kaiming, 2015) which scales weights by
    sqrt(2 / fan_in) to keep variance stable through ReLU-family activations.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    output_dim : int
        Number of output features (neurons).
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim

        # He initialisation: scale by sqrt(2/fan_in) for ReLU-family stability
        init_W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        # Bias initialised to zero (standard practice)
        init_b = np.zeros((1, output_dim))

        # Wrap in Parameter objects so the optimizer can find and update them
        self.W: Parameter = Parameter("W", init_W)
        self.b: Parameter = Parameter("b", init_b)

        # Cache for the input (needed during backward)
        self._input: np.ndarray | None = None

    # -- serialisation helpers (used by io.py) --
    @property
    def config(self) -> dict:
        """Return a dict describing this layer's constructor arguments."""
        return {"type": "Dense", "input_dim": self.input_dim, "output_dim": self.output_dim}

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Compute the linear transformation X @ W + b.

        Parameters
        ----------
        X : np.ndarray, shape (batch, input_dim)
            Input activations from the previous layer.
        training : bool
            Unused here; accepted for API consistency with Dropout.

        Returns
        -------
        np.ndarray, shape (batch, output_dim)
        """
        # Store input for use in backward pass (needed to compute dW)
        self._input = X

        # Linear transformation: matrix multiply + bias broadcast
        return X @ self.W.data + self.b.data

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Compute gradients and propagate the error signal backward.

        Math (for a single sample, then averaged over the batch):
            dL/dW  = X^T @ d_out          (chain rule through X @ W)
            dL/db  = sum(d_out, axis=0)    (bias gradient sums over batch)
            dL/dX  = d_out @ W^T           (propagate to previous layer)

        Parameters
        ----------
        d_out : np.ndarray, shape (batch, output_dim)
            Gradient of the loss w.r.t. this layer's output.

        Returns
        -------
        np.ndarray, shape (batch, input_dim)
            Gradient of the loss w.r.t. this layer's input.
        """
        X = self._input  # Retrieve cached input

        # Gradient w.r.t. weights: X^T @ d_out (shape: input_dim × output_dim)
        self.W.grad = X.T @ d_out

        # Gradient w.r.t. bias: sum over batch dimension (shape: 1 × output_dim)
        self.b.grad = np.sum(d_out, axis=0, keepdims=True)

        # Gradient w.r.t. input: propagate error backward through the weights
        d_input = d_out @ self.W.data.T

        return d_input

    def parameters(self) -> List[Parameter]:
        """Return all learnable parameters in this layer."""
        return [self.W, self.b]


# ---------------------------------------------------------------------------
# Dropout — regularisation via random neuron masking
# ---------------------------------------------------------------------------
class Dropout:
    """Inverted dropout: randomly zeros elements during training and scales
    survivors by 1/(1-p) so that expected values stay the same at test time.

    At inference (training=False), this layer is the identity function.

    Parameters
    ----------
    p : float
        Probability of dropping (zeroing) each element. Default 0.5.
    """

    def __init__(self, p: float = 0.5) -> None:
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p: float = p

        # Cache for the binary mask (needed during backward)
        self._mask: np.ndarray | None = None

    @property
    def config(self) -> dict:
        """Return a dict describing this layer's constructor arguments."""
        return {"type": "Dropout", "p": self.p}

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply inverted dropout during training; identity during inference.

        Parameters
        ----------
        X : np.ndarray
            Input activations.
        training : bool
            If True, apply dropout. If False, pass through unchanged.

        Returns
        -------
        np.ndarray
            Masked (and scaled) activations during training, or X unchanged.
        """
        if not training or self.p == 0.0:
            # During inference (or if p=0), pass input through unchanged
            return X

        # Generate binary mask: each element kept with probability (1 - p)
        self._mask = (np.random.rand(*X.shape) >= self.p).astype(X.dtype)

        # Scale by 1/(1-p) so expected value is unchanged ("inverted" dropout)
        # This avoids needing to scale at test time
        return X * self._mask / (1.0 - self.p)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Backpropagate through the dropout mask.

        Gradient flows only through the elements that were kept during forward,
        scaled by the same 1/(1-p) factor.

        Parameters
        ----------
        d_out : np.ndarray
            Gradient from the layer above.

        Returns
        -------
        np.ndarray
            Gradient w.r.t. the input.
        """
        if self._mask is None:
            # If forward was called in eval mode, mask is None → identity backward
            return d_out

        # Gradient flows only through kept elements, scaled by 1/(1-p)
        return d_out * self._mask / (1.0 - self.p)

    def parameters(self) -> List[Parameter]:
        """Dropout has no learnable parameters."""
        return []
