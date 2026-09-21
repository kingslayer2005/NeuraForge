"""
datasets.py — Load and cache datasets for NeuraForge benchmarks.

Provides two-moons, spirals, MNIST, and Fashion-MNIST.
Downloads from OpenML via scikit-learn and caches as .npz in data/.
"""

import os
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml, make_moons


def get_data_dir() -> Path:
    """Return the data directory, creating it if needed."""
    # Ensure we resolve relative to this file's location or the project root
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def make_spirals(n_samples: int = 1000, noise: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Generate a two-spiral dataset for decision boundary visualization.
    
    Ported from standard toy dataset implementations.
    """
    n = n_samples // 2
    # Generate angles for the two spirals
    theta = np.sqrt(np.random.rand(n)) * 2 * np.pi
    
    # Radius increases with angle
    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + np.random.randn(n, 2) * noise
    
    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    x_b = data_b + np.random.randn(n, 2) * noise
    
    res_a = np.append(x_a, np.zeros((n, 1)), axis=1)
    res_b = np.append(x_b, np.ones((n, 1)), axis=1)
    
    res = np.append(res_a, res_b, axis=0)
    np.random.shuffle(res)
    
    return res[:, :2].astype(np.float64), res[:, 2].astype(np.int64)


def load_toy_datasets(n_samples: int = 1000) -> dict:
    """Load two-moons and spirals datasets.
    
    Returns
    -------
    dict
        {"moons": (X, y), "spirals": (X, y)}
    """
    X_moons, y_moons = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    X_spirals, y_spirals = make_spirals(n_samples=n_samples, noise=0.5)
    
    return {
        "moons": (X_moons.astype(np.float64), y_moons.astype(np.int64)),
        "spirals": (X_spirals, y_spirals)
    }


def fetch_and_cache_openml(name: str, version: int, cache_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Fetch from OpenML and cache locally as .npz to avoid re-downloading."""
    data_dir = get_data_dir()
    cache_path = data_dir / f"{cache_name}.npz"
    
    if cache_path.exists():
        print(f"Loading {cache_name} from cache...")
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"]
        
    print(f"Downloading {cache_name} from OpenML... (this may take a minute)")
    # Download dataset
    dataset = fetch_openml(name, version=version, as_frame=False, parser="auto")
    X = dataset.data.astype(np.float64)
    y = dataset.target.astype(np.int64)
    
    print(f"Caching {cache_name} to {cache_path}...")
    np.savez_compressed(cache_path, X=X, y=y)
    
    return X, y


def load_mnist() -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST dataset (70,000 samples, 784 features, 10 classes)."""
    return fetch_and_cache_openml("mnist_784", version=1, cache_name="mnist")


def load_fashion_mnist() -> tuple[np.ndarray, np.ndarray]:
    """Load Fashion-MNIST dataset (70,000 samples, 784 features, 10 classes)."""
    return fetch_and_cache_openml("Fashion-MNIST", version=1, cache_name="fashion_mnist")
