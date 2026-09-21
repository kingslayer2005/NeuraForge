"""
seed.py — Deterministic seeding for reproducibility.

Sets both NumPy and Python stdlib random seeds so that every run
with the same seed produces identical results.
"""

import random  # Python's built-in random module
from typing import Optional

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for NumPy and Python stdlib for full reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to use. Default is 42.
    """
    # Set the NumPy global random seed (controls all np.random calls)
    np.random.seed(seed)

    # Set the Python stdlib random seed (controls random.random, etc.)
    random.seed(seed)
