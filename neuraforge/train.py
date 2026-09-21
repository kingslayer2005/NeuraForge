"""
train.py — Training loops and evaluation utilities.
"""

from typing import Any, Dict, Tuple

import numpy as np

from neuraforge.data import DataLoader
from neuraforge.io import get_weights, set_weights


class EarlyStopping:
    """Stops training when validation loss stops improving.

    Saves the weights of the best epoch and restores them upon stopping
    (or at the end of training).

    Parameters
    ----------
    patience : int
        Number of epochs to wait for improvement before stopping. Default 10.
    min_delta : float
        Minimum change in loss to qualify as an improvement. Default 0.0.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.best_weights: Dict[str, np.ndarray] | None = None

    def __call__(self, val_loss: float, model: Any) -> bool:
        """Check if training should stop.

        Parameters
        ----------
        val_loss : float
            Current validation loss.
        model : Sequential
            The model being trained (used to save best weights).

        Returns
        -------
        bool
            True if training should stop, False otherwise.
        """
        # Check if loss improved by at least min_delta
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save a deep copy of the current best weights
            self.best_weights = get_weights(model)
            return False

        self.counter += 1
        if self.counter >= self.patience:
            return True
        return False

    def restore_best_weights(self, model: Any) -> None:
        """Restore the model to the state with the best validation loss."""
        if self.best_weights is not None:
            set_weights(model, self.best_weights)


def accuracy_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute classification accuracy.

    Assumes targets are one-hot encoded or class indices.
    If predictions are 1D (binary), rounds them.
    """
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        # Multi-class: argmax over class dimension
        preds = np.argmax(predictions, axis=1)
        # If targets are one-hot, argmax them too
        if targets.ndim > 1 and targets.shape[1] > 1:
            targs = np.argmax(targets, axis=1)
        else:
            targs = targets
    else:
        # Binary: round probabilities to 0 or 1
        preds = np.round(predictions).flatten()
        targs = targets.flatten()

    return float(np.mean(preds == targs))


def evaluate(
    model: Any,
    loss_fn: Any,
    data_loader: DataLoader,
) -> Tuple[float, float]:
    """Evaluate the model on a dataset.

    Parameters
    ----------
    model : Sequential
        The neural network.
    loss_fn : Any
        The loss function object.
    data_loader : DataLoader
        Iterator yielding (X_batch, y_batch).

    Returns
    -------
    tuple of (loss, accuracy)
    """
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    for X_batch, y_batch in data_loader:
        batch_size = X_batch.shape[0]

        # Forward pass in eval mode (e.g. no dropout)
        preds = model.forward(X_batch, training=False)
        loss = loss_fn.forward(preds, y_batch)

        # loss_fn returns mean batch loss; multiply by batch size for total
        total_loss += loss * batch_size
        total_samples += batch_size

        all_preds.append(preds)
        all_targets.append(y_batch)

    # Compute epoch-level metrics
    epoch_loss = total_loss / total_samples
    
    cat_preds = np.vstack(all_preds)
    cat_targets = np.vstack(all_targets)
    epoch_acc = accuracy_score(cat_preds, cat_targets)

    return epoch_loss, epoch_acc


def fit(
    model: Any,
    optimizer: Any,
    loss_fn: Any,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 100,
    early_stopping: EarlyStopping | None = None,
    verbose: bool = True,
) -> Dict[str, list]:
    """Train the model for a given number of epochs.

    Parameters
    ----------
    model : Sequential
        The neural network.
    optimizer : Any
        Optimizer object (e.g. SGD, Adam).
    loss_fn : Any
        Loss function object.
    train_loader : DataLoader
        Training data iterator.
    val_loader : DataLoader, optional
        Validation data iterator.
    epochs : int
        Maximum number of epochs to train.
    early_stopping : EarlyStopping, optional
        Early stopping callback.
    verbose : bool
        If True, print progress every 10 epochs.

    Returns
    -------
    dict
        History dictionary containing lists of train/val metrics per epoch.
    """
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []

        # -- Training loop --
        for X_batch, y_batch in train_loader:
            batch_size = X_batch.shape[0]

            # 1. Zero gradients
            optimizer.zero_grad()

            # 2. Forward pass (training mode)
            preds = model.forward(X_batch, training=True)
            
            # 3. Compute loss
            loss = loss_fn.forward(preds, y_batch)

            # 4. Backward pass
            d_out = loss_fn.backward()
            model.backward(d_out)

            # 5. Optimizer step
            optimizer.step()

            # Accumulate metrics
            model_loss += loss * batch_size
            total_samples += batch_size
            all_preds.append(preds)
            all_targets.append(y_batch)

        # Compute training metrics for the epoch
        train_loss = model_loss / total_samples
        cat_preds = np.vstack(all_preds)
        cat_targets = np.vstack(all_targets)
        train_acc = accuracy_score(cat_preds, cat_targets)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # -- Validation loop --
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, loss_fn, val_loader)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Check early stopping
            if early_stopping is not None:
                if early_stopping(val_loss, model):
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch}")
                    break
        else:
            val_loss, val_acc = None, None
            # If no val set, early stopping uses train loss
            if early_stopping is not None:
                if early_stopping(train_loss, model):
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch}")
                    break

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            msg = f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
            if val_loss is not None:
                msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            print(msg)

    # Restore best weights if early stopping was used
    if early_stopping is not None:
        if verbose:
            print("Restoring best weights from early stopping.")
        early_stopping.restore_best_weights(model)

    return history
