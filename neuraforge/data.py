"""
data.py — Data loading, splitting, and preprocessing utilities.

Features:
- train_val_test_split: Shuffle and split arrays.
- Standardizer: Fit on training data only, transform all splits.
- DataLoader: Iterate over arrays in mini-batches.
"""

from typing import Dict, Tuple

import numpy as np


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Shuffle and split arrays into train, validation, and test sets.

    Parameters
    ----------
    X : np.ndarray
        Input features, shape (N, ...).
    y : np.ndarray
        Target labels/values, shape (N, ...).
    val_frac : float
        Fraction of data for the validation set.
    test_frac : float
        Fraction of data for the test set.
    seed : int
        Random seed for shuffling.

    Returns
    -------
    dict
        Dictionary containing 'X_train', 'Y_train', 'X_val', 'Y_val',
        'X_test', 'Y_test'.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    # Use a local RandomState so we don't interfere with global numpy RNG
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    n_test = int(n_samples * test_frac)
    n_val = int(n_samples * val_frac)
    n_train = n_samples - n_val - n_test

    # Slice indices for each split
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return {
        "X_train": X[train_idx],
        "Y_train": y[train_idx],
        "X_val": X[val_idx],
        "Y_val": y[val_idx],
        "X_test": X[test_idx],
        "Y_test": y[test_idx],
    }


class Standardizer:
    """Standardize features by removing the mean and scaling to unit variance.

    Centring and scaling happen independently on each feature.
    Crucially, statistics must be fitted ONLY on the training data to avoid
    data leakage.
    """

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.eps: float = 1e-8

    def fit(self, X: np.ndarray) -> None:
        """Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : np.ndarray
            Training data used to compute the mean and standard deviation.
        """
        # Axis 0 is the batch dimension; we compute stats per feature
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Perform standardization by centering and scaling.

        Parameters
        ----------
        X : np.ndarray
            Data to standardize.

        Returns
        -------
        np.ndarray
            Standardized data.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer must be fitted before calling transform.")
        return (X - self.mean) / (self.std + self.eps)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it.

        Parameters
        ----------
        X : np.ndarray
            Training data.

        Returns
        -------
        np.ndarray
            Transformed array.
        """
        self.fit(X)
        return self.transform(X)


class DataLoader:
    """An iterator that yields mini-batches of data.

    Parameters
    ----------
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target labels.
    batch_size : int
        Number of samples per batch. Default 32.
    shuffle : bool
        Whether to shuffle the data at the start of each epoch. Default True.
    seed : int | None
        Seed for shuffling. If None, uses global numpy state.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.X: np.ndarray = X
        self.y: np.ndarray = y
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.n_samples: int = X.shape[0]

        # Use a local RNG for predictable shuffling independent of global state
        self.rng = np.random.RandomState(seed)

        self._indices = np.arange(self.n_samples)
        self._current_idx: int = 0

    def __iter__(self) -> "DataLoader":
        """Reset the iterator and shuffle if requested."""
        self._current_idx = 0
        if self.shuffle:
            self.rng.shuffle(self._indices)
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """Yield the next mini-batch.

        Returns
        -------
        tuple of (X_batch, y_batch)
        """
        if self._current_idx >= self.n_samples:
            raise StopIteration

        # Slice the next batch of indices
        end_idx = min(self._current_idx + self.batch_size, self.n_samples)
        batch_indices = self._indices[self._current_idx : end_idx]
        self._current_idx = end_idx

        # Return the corresponding data arrays
        return self.X[batch_indices], self.y[batch_indices]

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        # Ceiling division for total number of batches
        return int(np.ceil(self.n_samples / self.batch_size))
