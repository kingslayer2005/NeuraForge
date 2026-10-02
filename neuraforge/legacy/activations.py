"""
activations.py — Activation functions for NeuraForge.

Every activation follows the same protocol:
    - forward(Z, training=True)  → output array
    - backward(d_out)            → gradient w.r.t. the activation's input
    - parameters()               → list of Parameter objects (empty for most)
    - config                     → dict describing constructor arguments

Honest framing:
    - When alpha=0, ForageAct is exactly SiLU/Swish (Ramachandran et al., 2017).
      The alpha*tanh(x) term is a learnable extension.
    - SiLU is x * sigmoid(x), proposed independently by Elfwing et al. (2018)
      and Ramachandran et al. (2017) as Swish.
    - GELU uses the tanh approximation from Hendrycks & Gimpel (2016).
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from neuraforge.layers import Parameter


# ---------------------------------------------------------------------------
# ReLU — Rectified Linear Unit
# ---------------------------------------------------------------------------
class ReLU:
    """ReLU(z) = max(0, z).

    The most common activation. Simple gradient: 1 where z > 0, 0 elsewhere.
    Suffers from "dying ReLU" where neurons that output 0 stop learning.
    """

    @property
    def config(self) -> dict:
        return {"type": "ReLU"}

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply ReLU element-wise: max(0, z).

        Parameters
        ----------
        Z : np.ndarray
            Pre-activation values.
        training : bool
            Unused; accepted for API consistency.

        Returns
        -------
        np.ndarray
            Activated values, same shape as Z.
        """
        # Cache input for backward (need to know where z > 0)
        self._Z = Z
        # Element-wise maximum with zero
        return np.maximum(0, Z)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Backpropagate through ReLU.

        Gradient rule: dL/dZ = dL/d_out * indicator(Z > 0)
        Gradient is 1 where Z was positive, 0 where Z was ≤ 0.

        Parameters
        ----------
        d_out : np.ndarray
            Gradient from the layer above.

        Returns
        -------
        np.ndarray
            Gradient w.r.t. the pre-activation input Z.
        """
        # Gradient passes through only where the input was positive
        return d_out * (self._Z > 0).astype(d_out.dtype)

    def parameters(self) -> List[Parameter]:
        """ReLU has no learnable parameters."""
        return []


# ---------------------------------------------------------------------------
# LeakyReLU
# ---------------------------------------------------------------------------
class LeakyReLU:
    """LeakyReLU(z) = z if z > 0, else negative_slope * z.

    Avoids the "dying ReLU" problem by allowing a small gradient when z < 0.

    Parameters
    ----------
    negative_slope : float
        Slope for negative inputs. Default 0.01.
    """

    def __init__(self, negative_slope: float = 0.01) -> None:
        self.negative_slope: float = negative_slope

    @property
    def config(self) -> dict:
        return {"type": "LeakyReLU", "negative_slope": self.negative_slope}

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply LeakyReLU element-wise.

        Parameters
        ----------
        Z : np.ndarray
            Pre-activation values.
        training : bool
            Unused; accepted for API consistency.

        Returns
        -------
        np.ndarray
            Activated values.
        """
        # Cache input for backward
        self._Z = Z
        # Where z > 0 keep z; where z ≤ 0 scale by negative_slope
        return np.where(Z > 0, Z, self.negative_slope * Z)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Backpropagate through LeakyReLU.

        Gradient: 1 where Z > 0, negative_slope where Z ≤ 0.

        Parameters
        ----------
        d_out : np.ndarray
            Gradient from the layer above.

        Returns
        -------
        np.ndarray
            Gradient w.r.t. input Z.
        """
        # Gradient is 1 for positive inputs, negative_slope for non-positive
        grad_mask = np.where(self._Z > 0, 1.0, self.negative_slope)
        return d_out * grad_mask

    def parameters(self) -> List[Parameter]:
        """LeakyReLU has no learnable parameters."""
        return []


# ---------------------------------------------------------------------------
# Tanh
# ---------------------------------------------------------------------------
class Tanh:
    """Tanh activation: tanh(z) = (e^z - e^-z) / (e^z + e^-z).

    Outputs in range (-1, 1). Derivative: 1 - tanh²(z).
    """

    @property
    def config(self) -> dict:
        return {"type": "Tanh"}

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply tanh element-wise.

        Parameters
        ----------
        Z : np.ndarray
            Pre-activation values.
        training : bool
            Unused.

        Returns
        -------
        np.ndarray
            tanh(Z), values in (-1, 1).
        """
        # Compute and cache tanh for use in backward
        self._tanh = np.tanh(Z)
        return self._tanh

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Backpropagate through tanh.

        Gradient: dL/dZ = dL/d_out * (1 - tanh²(Z))

        Parameters
        ----------
        d_out : np.ndarray
            Gradient from the layer above.

        Returns
        -------
        np.ndarray
            Gradient w.r.t. input Z.
        """
        # Derivative of tanh is (1 - tanh²), using cached tanh value
        return d_out * (1.0 - self._tanh ** 2)

    def parameters(self) -> List[Parameter]:
        """Tanh has no learnable parameters."""
        return []


# ---------------------------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------------------------
class Sigmoid:
    """Sigmoid activation: σ(z) = 1 / (1 + e^(-z)).

    Outputs in range (0, 1). Derivative: σ(z) * (1 - σ(z)).
    """

    @property
    def config(self) -> dict:
        return {"type": "Sigmoid"}

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply sigmoid element-wise.

        Parameters
        ----------
        Z : np.ndarray
            Pre-activation values.
        training : bool
            Unused.

        Returns
        -------
        np.ndarray
            σ(Z), values in (0, 1).
        """
        # Compute and cache sigmoid for use in backward
        self._sigmoid = 1.0 / (1.0 + np.exp(-Z))
        return self._sigmoid

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Backpropagate through sigmoid.

        Gradient: dL/dZ = dL/d_out * σ(Z) * (1 - σ(Z))

        Parameters
        ----------
        d_out : np.ndarray
            Gradient from the layer above.

        Returns
        -------
        np.ndarray
            Gradient w.r.t. input Z.
        """
        # Derivative of sigmoid: σ * (1 - σ), using cached sigmoid
        return d_out * self._sigmoid * (1.0 - self._sigmoid)

    def parameters(self) -> List[Parameter]:
        """Sigmoid has no learnable parameters."""
        return []


# ---------------------------------------------------------------------------
# SiLU (Swish) — z * σ(z)
# ---------------------------------------------------------------------------
class SiLU:
    """SiLU (Sigmoid Linear Unit), also known as Swish.

    f(z) = z * σ(z), where σ is the sigmoid function.
    Proposed by Elfwing et al. (2018) and Ramachandran et al. (2017).

    Note: ForageAct with alpha=0 is exactly SiLU.
    """

    @property
    def config(self) -> dict:
        return {"type": "SiLU"}

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply SiLU element-wise: z * σ(z).

        Parameters
        ----------
        Z : np.ndarray
            Pre-activation values.
        training : bool
            Unused.

        Returns
        -------
        np.ndarray
            SiLU(Z).
        """
        # Cache both Z and σ(Z) for backward
        self._Z = Z
        self._sigmoid = 1.0 / (1.0 + np.exp(-Z))
        # SiLU = z * sigmoid(z)
        return Z * self._sigmoid

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Backpropagate through SiLU.

        Gradient of z·σ(z) w.r.t. z using the product rule:
            d/dz [z·σ(z)] = σ(z) + z·σ(z)·(1 - σ(z))
                           = σ(z) + z·σ'(z)

        Parameters
        ----------
        d_out : np.ndarray
            Gradient from the layer above.

        Returns
        -------
        np.ndarray
            Gradient w.r.t. input Z.
        """
        s = self._sigmoid   # σ(z)
        z = self._Z         # z

        # σ'(z) = σ(z) * (1 - σ(z))
        s_prime = s * (1.0 - s)

        # Product rule: d/dz[z·σ(z)] = σ(z) + z·σ'(z)
        grad = s + z * s_prime

        return d_out * grad

    def parameters(self) -> List[Parameter]:
        """SiLU has no learnable parameters."""
        return []


# ---------------------------------------------------------------------------
# GELU (tanh approximation) — Hendrycks & Gimpel (2016)
# ---------------------------------------------------------------------------
class GELU:
    """GELU activation using the tanh approximation.

    f(z) = 0.5 * z * (1 + tanh(sqrt(2/π) * (z + 0.044715 * z³)))

    This is the approximation used by GPT-2 and BERT. The exact GELU
    uses the Gaussian CDF, but the tanh form is faster and widely adopted.
    """

    # Precompute the constant sqrt(2/π) ≈ 0.7978845608
    _SQRT_2_OVER_PI: float = math.sqrt(2.0 / math.pi)
    # Coefficient for the cubic term in the tanh approximation
    _COEFF: float = 0.044715

    @property
    def config(self) -> dict:
        return {"type": "GELU"}

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply GELU (tanh approximation) element-wise.

        Parameters
        ----------
        Z : np.ndarray
            Pre-activation values.
        training : bool
            Unused.

        Returns
        -------
        np.ndarray
            GELU(Z).
        """
        # Cache Z for backward
        self._Z = Z

        # Inner argument of tanh: sqrt(2/π) * (z + 0.044715 * z³)
        self._inner = self._SQRT_2_OVER_PI * (Z + self._COEFF * Z ** 3)

        # Compute tanh of the inner argument
        self._tanh_inner = np.tanh(self._inner)

        # GELU = 0.5 * z * (1 + tanh(inner))
        return 0.5 * Z * (1.0 + self._tanh_inner)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Backpropagate through GELU (tanh approximation).

        Using chain rule on f(z) = 0.5·z·(1 + tanh(g(z))):
            f'(z) = 0.5·(1 + tanh(g)) + 0.5·z·(1 - tanh²(g))·g'(z)
        where g(z) = sqrt(2/π)·(z + 0.044715·z³)
          and g'(z) = sqrt(2/π)·(1 + 3·0.044715·z²)

        Parameters
        ----------
        d_out : np.ndarray
            Gradient from the layer above.

        Returns
        -------
        np.ndarray
            Gradient w.r.t. input Z.
        """
        z = self._Z
        t = self._tanh_inner  # tanh(g(z))

        # Derivative of the inner function g(z) w.r.t. z:
        # g'(z) = sqrt(2/π) * (1 + 3 * 0.044715 * z²)
        g_prime = self._SQRT_2_OVER_PI * (1.0 + 3.0 * self._COEFF * z ** 2)

        # sech²(g) = 1 - tanh²(g), derivative of tanh
        sech2 = 1.0 - t ** 2

        # Full derivative via product rule:
        # f'(z) = 0.5 * (1 + tanh(g)) + 0.5 * z * sech²(g) * g'(z)
        grad = 0.5 * (1.0 + t) + 0.5 * z * sech2 * g_prime

        return d_out * grad

    def parameters(self) -> List[Parameter]:
        """GELU has no learnable parameters."""
        return []


# ---------------------------------------------------------------------------
# ForageAct — trainable activation (NeuraForge's custom contribution)
# ---------------------------------------------------------------------------
class ForageAct:
    """ForageAct: f(z) = z·σ(z) + α·tanh(z).

    When α = 0, this reduces to SiLU/Swish (Ramachandran et al., 2017).
    The α·tanh(z) term adds a learnable bias toward odd-function behaviour,
    which may help when targets benefit from sign-sensitive activations.

    Three modes for α:
        - "fixed":      α is a constant (no gradient, not learned).
        - "scalar":     α is a single learnable scalar shared across all neurons.
        - "per_neuron": α is a learnable vector with one value per neuron.

    Parameters
    ----------
    mode : str
        One of "fixed", "scalar", "per_neuron".
    init_alpha : float
        Initial value for α. Default 0.1.
    n_neurons : int | None
        Required when mode="per_neuron". The number of neurons (output dim)
        for the preceding Dense layer.
    """

    def __init__(
        self,
        mode: str = "scalar",
        init_alpha: float = 0.1,
        n_neurons: int | None = None,
    ) -> None:
        if mode not in ("fixed", "scalar", "per_neuron"):
            raise ValueError(f"ForageAct mode must be 'fixed', 'scalar', or 'per_neuron', got {mode!r}")

        self.mode: str = mode
        self.init_alpha: float = init_alpha
        self.n_neurons: int | None = n_neurons

        if mode == "fixed":
            # Fixed alpha: store as plain ndarray but don't register as Parameter
            self._alpha_value: np.ndarray = np.array([init_alpha])
            self._alpha_param: Parameter | None = None

        elif mode == "scalar":
            # Learnable scalar alpha: shape (1,) so in-place update works
            alpha_data = np.array([init_alpha])
            self._alpha_param = Parameter("alpha", alpha_data)
            self._alpha_value = self._alpha_param.data  # Points to same array

        elif mode == "per_neuron":
            if n_neurons is None:
                raise ValueError("n_neurons is required when mode='per_neuron'")
            # Learnable per-neuron alpha: shape (n_neurons,)
            alpha_data = np.full(n_neurons, init_alpha)
            self._alpha_param = Parameter("alpha", alpha_data)
            self._alpha_value = self._alpha_param.data  # Points to same array

    @property
    def config(self) -> dict:
        return {
            "type": "ForageAct",
            "mode": self.mode,
            "init_alpha": self.init_alpha,
            "n_neurons": self.n_neurons,
        }

    @property
    def alpha(self) -> np.ndarray:
        """Current alpha value(s), always as ndarray."""
        if self._alpha_param is not None:
            return self._alpha_param.data
        return self._alpha_value

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply ForageAct: f(z) = z·σ(z) + α·tanh(z).

        Parameters
        ----------
        Z : np.ndarray, shape (batch, features)
            Pre-activation values.
        training : bool
            Unused (ForageAct behaves the same in train and eval).

        Returns
        -------
        np.ndarray, shape (batch, features)
            Activated values.
        """
        # Cache Z, σ(Z), and tanh(Z) for use in backward
        self._Z = Z
        self._sigmoid = 1.0 / (1.0 + np.exp(-Z))  # σ(z)
        self._tanh = np.tanh(Z)                     # tanh(z)

        # f(z) = z·σ(z) + α·tanh(z)
        # α broadcasts: scalar (1,) or per-neuron (features,) both work
        return Z * self._sigmoid + self.alpha * self._tanh

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Backpropagate through ForageAct.

        Gradient of f(z) = z·σ(z) + α·tanh(z) w.r.t. z:
            df/dz = σ(z) + z·σ'(z) + α·(1 - tanh²(z))
        where σ'(z) = σ(z)·(1 - σ(z)).

        Gradient w.r.t. α:
            df/dα = tanh(z)
        Summed over the batch (scalar mode) or over the batch only (per-neuron mode).

        Parameters
        ----------
        d_out : np.ndarray, shape (batch, features)
            Gradient from the layer above.

        Returns
        -------
        np.ndarray, shape (batch, features)
            Gradient w.r.t. input Z.
        """
        s = self._sigmoid   # σ(z), shape (batch, features)
        z = self._Z         # z, shape (batch, features)
        t = self._tanh      # tanh(z), shape (batch, features)
        alpha = self.alpha   # shape (1,) or (features,)

        # σ'(z) = σ(z) · (1 - σ(z))
        s_prime = s * (1.0 - s)

        # d/dz [z·σ(z)] = σ(z) + z·σ'(z)  (product rule)
        d_swish = s + z * s_prime

        # d/dz [tanh(z)] = 1 - tanh²(z)
        d_tanh = 1.0 - t ** 2

        # Total gradient w.r.t. z:
        # df/dz = d_swish + α·d_tanh
        dZ = d_out * (d_swish + alpha * d_tanh)

        # Gradient w.r.t. α (only if learnable)
        if self._alpha_param is not None:
            if self.mode == "scalar":
                # Sum over all elements (batch × features) → scalar in shape (1,)
                self._alpha_param.grad = np.array([np.sum(d_out * t)])
            elif self.mode == "per_neuron":
                # Sum over batch dimension only → shape (features,)
                self._alpha_param.grad = np.sum(d_out * t, axis=0)

        return dZ

    def parameters(self) -> List[Parameter]:
        """Return alpha as a learnable parameter (empty list if mode='fixed')."""
        if self._alpha_param is not None:
            return [self._alpha_param]
        return []
