"""
NeuraForge: Neural network framework from scratch in pure NumPy.
"""

__version__ = "0.1.0"

# Expose core primitives at the package level for clean imports
from neuraforge.activations import (
    GELU,
    LeakyReLU,
    ReLU,
    Sigmoid,
    SiLU,
    Tanh,
    ForageAct,
)
from neuraforge.layers import Dense, Dropout
from neuraforge.losses import BCEWithLogitsLoss, MSELoss, SoftmaxCrossEntropy
from neuraforge.model import Sequential
from neuraforge.optimizers import SGD, Adam, MomentumSGD, NeuroGrad
from neuraforge.seed import seed_everything
