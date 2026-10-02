"""
optimizers.py — Gradient-based optimizers for NeuraForge.

All optimizers follow the same protocol:
    - __init__(params, lr, ...)   where params is a list of Parameter objects
    - step()                      updates every parameter's .data in place
    - zero_grad()                 resets every parameter's .grad to None

Honest framing:
    - NeuroGrad uses EMA momentum: v = β·v + (1−β)·g.
      Without clipping, this is mathematically identical to classical momentum
      v_c = β·v_c + g with learning rate lr_c = lr * (1 − β).
      Any difference in benchmarks comes from the gradient clipping or the
      separately tuned learning rate.
    - Classical momentum (MomentumSGD) uses v = β·v + g.
    - Gradient clipping follows Pascanu et al. (2013), "On the difficulty of
      training recurrent neural networks."
"""

from __future__ import annotations

import numpy as np

from neuraforge.layers import Parameter


# ---------------------------------------------------------------------------
# SGD — Vanilla Stochastic Gradient Descent
# ---------------------------------------------------------------------------
class SGD:
    """Vanilla stochastic gradient descent: p ← p − lr · g.

    No momentum, no state. The simplest possible optimizer.

    Parameters
    ----------
    params : list[Parameter]
        Learnable parameters from model.parameters().
    lr : float
        Learning rate. Default 0.01.
    """

    def __init__(self, params: list[Parameter], lr: float = 0.01) -> None:
        self.params: list[Parameter] = params
        self.lr: float = lr

    def step(self) -> None:
        """Update every parameter by subtracting lr * gradient in place."""
        for p in self.params:
            if p.grad is None:
                # Skip parameters that didn't receive a gradient this step
                continue
            # Vanilla SGD update: p ← p − lr · g
            p.data -= self.lr * p.grad

    def zero_grad(self) -> None:
        """Reset all gradients to None before the next forward/backward pass."""
        for p in self.params:
            p.grad = None


# ---------------------------------------------------------------------------
# MomentumSGD — Classical momentum
# ---------------------------------------------------------------------------
class MomentumSGD:
    """SGD with classical momentum: v ← β·v + g,  p ← p − lr · v.

    Classical (Polyak) momentum accumulates past gradients in a velocity
    vector. Unlike EMA momentum, the gradient is added without scaling
    by (1 − β), so the effective step size grows as velocity builds up.

    Parameters
    ----------
    params : list[Parameter]
        Learnable parameters from model.parameters().
    lr : float
        Learning rate. Default 0.01.
    beta : float
        Momentum coefficient. Default 0.9.
    """

    def __init__(
        self,
        params: list[Parameter],
        lr: float = 0.01,
        beta: float = 0.9,
    ) -> None:
        self.params: list[Parameter] = params
        self.lr: float = lr
        self.beta: float = beta

        # Initialise velocity buffers to zero for each parameter
        self._velocities: list[np.ndarray] = [
            np.zeros_like(p.data) for p in params
        ]

    def step(self) -> None:
        """Update every parameter using classical momentum."""
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            v = self._velocities[i]

            # Classical momentum: v ← β·v + g (no (1−β) scaling)
            v[:] = self.beta * v + p.grad

            # Parameter update: p ← p − lr · v
            p.data -= self.lr * v

    def zero_grad(self) -> None:
        """Reset all gradients to None."""
        for p in self.params:
            p.grad = None


# ---------------------------------------------------------------------------
# Adam — Adaptive Moment Estimation (Kingma & Ba, 2015)
# ---------------------------------------------------------------------------
class Adam:
    """Adam optimizer with bias correction.

    Maintains per-parameter running averages of the first moment (mean)
    and second moment (uncentred variance) of the gradients, with bias
    correction to account for the zero-initialised estimates.

    Parameters
    ----------
    params : list[Parameter]
        Learnable parameters from model.parameters().
    lr : float
        Learning rate. Default 0.001.
    beta1 : float
        Decay rate for the first moment estimate. Default 0.9.
    beta2 : float
        Decay rate for the second moment estimate. Default 0.999.
    eps : float
        Small constant for numerical stability. Default 1e-8.
    """

    def __init__(
        self,
        params: list[Parameter],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.params: list[Parameter] = params
        self.lr: float = lr
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.eps: float = eps

        # First moment estimate (mean of gradients), initialised to zero
        self._m: list[np.ndarray] = [np.zeros_like(p.data) for p in params]
        # Second moment estimate (mean of squared gradients), initialised to zero
        self._v: list[np.ndarray] = [np.zeros_like(p.data) for p in params]
        # Timestep counter (for bias correction)
        self._t: int = 0

    def step(self) -> None:
        """Update every parameter using bias-corrected Adam."""
        # Increment timestep (starts at 1 on first call)
        self._t += 1

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad  # Current gradient

            # Update biased first moment estimate: m ← β1·m + (1−β1)·g
            self._m[i][:] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g

            # Update biased second moment estimate: v ← β2·v + (1−β2)·g²
            self._v[i][:] = self.beta2 * self._v[i] + (1.0 - self.beta2) * g ** 2

            # Bias correction: compensate for zero-initialised moments
            # m̂ = m / (1 − β1^t)
            m_hat = self._m[i] / (1.0 - self.beta1 ** self._t)
            # v̂ = v / (1 − β2^t)
            v_hat = self._v[i] / (1.0 - self.beta2 ** self._t)

            # Parameter update: p ← p − lr · m̂ / (√v̂ + ε)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        """Reset all gradients to None."""
        for p in self.params:
            p.grad = None


# ---------------------------------------------------------------------------
# NeuroGrad — EMA momentum + gradient clipping (NeuraForge's custom optimizer)
# ---------------------------------------------------------------------------
class NeuroGrad:
    """NeuroGrad: EMA momentum with per-tensor or global gradient-norm clipping.

    Update rule:
        1. Clip gradient g by its norm (per-tensor or global).
        2. Update velocity: v ← β·v + (1−β)·g_clipped   (EMA form).
        3. Update parameter: p ← p − lr · v.

    Honest note: without clipping, EMA momentum v = β·v + (1−β)·g is
    mathematically identical to classical momentum v_c = β·v_c + g with
    learning rate lr_c = lr * (1 − β). Any difference observed in
    benchmarks comes from:
        (a) the gradient clipping, and/or
        (b) separately tuned learning rates.

    Gradient clipping follows Pascanu et al. (2013).

    Parameters
    ----------
    params : list[Parameter]
        Learnable parameters from model.parameters().
    lr : float
        Learning rate. Default 0.01.
    beta : float
        EMA decay rate for velocity. Default 0.9.
    clip_value : float
        Maximum allowed gradient norm. Default 1.0.
    clip_mode : str
        "per_tensor" clips each parameter's gradient independently.
        "global_norm" computes a single norm across all parameters and
        rescales all gradients by the same factor if it exceeds clip_value.
    """

    def __init__(
        self,
        params: list[Parameter],
        lr: float = 0.01,
        beta: float = 0.9,
        clip_value: float = 1.0,
        clip_mode: str = "per_tensor",
    ) -> None:
        if clip_mode not in ("per_tensor", "global_norm", "none"):
            raise ValueError(f"clip_mode must be 'per_tensor', 'global_norm', or 'none', got {clip_mode!r}")

        self.params: list[Parameter] = params
        self.lr: float = lr
        self.beta: float = beta
        self.clip_value: float = clip_value
        self.clip_mode: str = clip_mode

        # Initialise EMA velocity buffers to zero
        self._velocities: list[np.ndarray] = [
            np.zeros_like(p.data) for p in params
        ]

    def _clip_per_tensor(self, grad: np.ndarray) -> np.ndarray:
        """Clip a single gradient tensor by its L2 norm.

        If ||g|| > clip_value, rescale: g ← g * (clip_value / ||g||).

        Parameters
        ----------
        grad : np.ndarray
            The gradient tensor to clip.

        Returns
        -------
        np.ndarray
            Clipped gradient (may be the same object if no clipping needed).
        """
        norm = np.linalg.norm(grad)
        if norm > self.clip_value:
            # Scale gradient down so its norm equals clip_value
            grad = grad * (self.clip_value / norm)
        return grad

    def step(self) -> None:
        """Update every parameter using clipped-EMA-momentum."""
        if self.clip_mode == "global_norm":
            # Compute the global gradient norm across all parameters
            total_norm_sq = 0.0
            for p in self.params:
                if p.grad is not None:
                    # Sum of squared norms: ||g_1||² + ||g_2||² + ...
                    total_norm_sq += np.sum(p.grad ** 2)
            global_norm = np.sqrt(total_norm_sq)

            # Compute the scaling factor (1.0 if no clipping needed)
            if global_norm > self.clip_value:
                clip_scale = self.clip_value / global_norm
            else:
                clip_scale = 1.0

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad  # Current gradient

            # Step 1: clip the gradient
            if self.clip_mode == "per_tensor":
                g = self._clip_per_tensor(g)
            elif self.clip_mode == "global_norm":
                g = g * clip_scale  # Same scale factor for all params
            # else clip_mode == "none": no clipping

            v = self._velocities[i]

            # Step 2: EMA velocity update: v ← β·v + (1−β)·g
            v[:] = self.beta * v + (1.0 - self.beta) * g

            # Step 3: parameter update: p ← p − lr · v
            p.data -= self.lr * v

    def zero_grad(self) -> None:
        """Reset all gradients to None."""
        for p in self.params:
            p.grad = None
