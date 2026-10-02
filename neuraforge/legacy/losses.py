"""
losses.py — Loss functions for NeuraForge.

Every loss follows the same protocol:
    - forward(predictions, targets) → scalar loss value
    - backward()                    → gradient w.r.t. the predictions

All losses return the mean loss over the batch (averaged by N = batch size).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Softmax + Cross-Entropy (for multi-class classification)
# ---------------------------------------------------------------------------
class SoftmaxCrossEntropy:
    """Combined softmax activation and cross-entropy loss.

    Combining softmax and cross-entropy into one layer gives a very clean
    gradient: dL/d(logits) = (probs - Y_onehot) / N, which avoids
    numerical issues from computing log(softmax) separately.

    Expects raw logits (not probabilities) and one-hot encoded targets.
    """

    @property
    def config(self) -> dict:
        return {"type": "SoftmaxCrossEntropy"}

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Compute the softmax cross-entropy loss.

        Parameters
        ----------
        logits : np.ndarray, shape (N, C)
            Raw (unnormalized) scores for each class.
        targets : np.ndarray, shape (N, C)
            One-hot encoded ground-truth labels.

        Returns
        -------
        float
            Mean cross-entropy loss over the batch.
        """
        # Subtract the row-wise max for numerical stability (log-sum-exp trick).
        # This prevents overflow in exp() without changing the softmax result.
        shifted = logits - np.max(logits, axis=1, keepdims=True)

        # Compute softmax probabilities: exp(z_i) / Σ_j exp(z_j)
        exp_shifted = np.exp(shifted)
        self._probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

        # Cache targets for backward
        self._targets = targets
        # Cache batch size for averaging
        self._N = targets.shape[0]

        # Cross-entropy: -Σ y_i · log(p_i), averaged over batch
        # Add small epsilon (1e-9) inside log to avoid log(0)
        loss = -np.sum(targets * np.log(self._probs + 1e-9)) / self._N

        return float(loss)

    def backward(self) -> np.ndarray:
        """Compute gradient of cross-entropy loss w.r.t. logits.

        The combined softmax + cross-entropy gradient simplifies to:
            dL/d(logits) = (softmax_probs - one_hot_targets) / N

        This elegant form is why we combine softmax and CE into one layer.

        Returns
        -------
        np.ndarray, shape (N, C)
            Gradient w.r.t. the raw logits.
        """
        # (probs - targets) / N  — the classic softmax-CE gradient
        return (self._probs - self._targets) / self._N


# ---------------------------------------------------------------------------
# Mean Squared Error (for regression)
# ---------------------------------------------------------------------------
class MSELoss:
    """Mean Squared Error loss: L = mean((predictions - targets)²).

    Used for regression tasks.
    """

    @property
    def config(self) -> dict:
        return {"type": "MSELoss"}

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute the mean squared error.

        Parameters
        ----------
        predictions : np.ndarray, shape (N, D)
            Model outputs.
        targets : np.ndarray, shape (N, D)
            Ground-truth values.

        Returns
        -------
        float
            Mean squared error averaged over all elements.
        """
        # Cache for backward
        self._predictions = predictions
        self._targets = targets
        self._N = predictions.shape[0]

        # MSE = mean of (pred - target)² over all elements
        # We use the total number of elements for the mean
        loss = np.mean((predictions - targets) ** 2)
        return float(loss)

    def backward(self) -> np.ndarray:
        """Compute gradient of MSE w.r.t. predictions.

        dL/d(pred) = 2 * (pred - target) / (N * D)
        where N*D is the total number of elements (to match np.mean in forward).

        Returns
        -------
        np.ndarray, shape (N, D)
            Gradient w.r.t. predictions.
        """
        # Total number of elements for the mean
        total = self._predictions.size

        # Gradient: 2 * (pred - target) / total_elements
        return 2.0 * (self._predictions - self._targets) / total


# ---------------------------------------------------------------------------
# Binary Cross-Entropy with Logits (for binary classification)
# ---------------------------------------------------------------------------
class BCEWithLogitsLoss:
    """Numerically stable binary cross-entropy computed from raw logits.

    Uses the identity:
        BCE = max(x, 0) - x·y + log(1 + exp(-|x|))
    which avoids overflow from exp(x) when x is large positive,
    and avoids overflow from exp(-x) when x is large negative.

    Expects raw logits (not probabilities) and binary targets in {0, 1}.
    """

    @property
    def config(self) -> dict:
        return {"type": "BCEWithLogitsLoss"}

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Compute the numerically stable binary cross-entropy loss.

        Parameters
        ----------
        logits : np.ndarray, shape (N, D)
            Raw (unnormalized) scores.
        targets : np.ndarray, shape (N, D)
            Binary ground-truth labels (0 or 1).

        Returns
        -------
        float
            Mean binary cross-entropy loss.
        """
        # Cache for backward
        self._logits = logits
        self._targets = targets

        # Numerically stable formulation:
        # max(x, 0) - x·y + log(1 + exp(-|x|))
        # This avoids overflow for both large positive and large negative logits
        relu_x = np.maximum(logits, 0)  # max(x, 0)
        loss = relu_x - logits * targets + np.log(1.0 + np.exp(-np.abs(logits)))

        # Average over all elements
        return float(np.mean(loss))

    def backward(self) -> np.ndarray:
        """Compute gradient of BCE w.r.t. logits.

        dL/d(logits) = (σ(logits) - targets) / (N * D)
        where σ is the sigmoid function.

        This has the same elegant form as the softmax-CE gradient
        but for the binary case.

        Returns
        -------
        np.ndarray, shape (N, D)
            Gradient w.r.t. logits.
        """
        # Total number of elements (to match the np.mean in forward)
        total = self._logits.size

        # σ(logits) — sigmoid of the raw logits
        sigmoid = 1.0 / (1.0 + np.exp(-self._logits))

        # Gradient: (σ(x) - y) / total_elements
        return (sigmoid - self._targets) / total
