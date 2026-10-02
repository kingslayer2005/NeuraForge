"""
NeuraForge: Neural network framework from scratch in pure NumPy.

Modules:
    autograd     — Reverse-mode automatic differentiation engine (Tensor class)
    nn           — Neural network layers built on autograd (Dense, Conv2d, BatchNorm, etc.)
    transformer  — Transformer components (MultiHeadAttention, TransformerBlock, etc.)
    layers       — Legacy layer implementations with hand-written backward
    activations  — Legacy activation functions
    losses       — Loss functions
    optimizers   — Parameter update algorithms (SGD, Adam, NeuroGrad)
    model        — Sequential container
    growth       — Net2WiderNet dynamic widening
    pruning      — Taylor-based structured pruning
"""

__version__ = "0.2.0"

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

# New autograd engine
from neuraforge.autograd import Tensor, no_grad

