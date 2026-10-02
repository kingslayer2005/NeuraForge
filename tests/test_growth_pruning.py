"""
test_growth_pruning.py — Tests for dynamic architecture changes.

Verifies that Net2WiderNet preserves the exact output of the network,
and that pruning correctly removes neurons and their connections without
crashing the forward or backward pass.
"""

import numpy as np
import pytest

from neuraforge.activations import ForageAct, ReLU
from neuraforge.growth import grow_layer
from neuraforge.layers import Dense
from neuraforge.model import Sequential
from neuraforge.pruning import prune_layer
from neuraforge.seed import seed_everything


@pytest.fixture(autouse=True)
def setup_seed():
    seed_everything(42)


def test_grow_layer_preserves_output():
    batch_size, in_dim, hidden_dim, out_dim = 2, 5, 10, 3
    X = np.random.randn(batch_size, in_dim).astype(np.float64)
    
    # Simple MLP with standard activation
    model1 = Sequential(
        Dense(in_dim, hidden_dim),
        ReLU(),
        Dense(hidden_dim, out_dim)
    )
    
    # Store original output
    orig_output = model1.forward(X)
    
    # Grow the first hidden layer by 5 neurons
    grow_layer(model1, layer_idx=0, num_new_neurons=5)
    
    # Ensure dimensions updated
    assert model1.layers[0].output_dim == hidden_dim + 5
    assert model1.layers[0].W.data.shape == (in_dim, hidden_dim + 5)
    assert model1.layers[2].input_dim == hidden_dim + 5
    assert model1.layers[2].W.data.shape == (hidden_dim + 5, out_dim)
    
    # Check that output is exactly preserved
    new_output = model1.forward(X)
    np.testing.assert_allclose(
        orig_output, new_output, 
        rtol=1e-6, atol=1e-8,
        err_msg="Net2WiderNet did not preserve output for ReLU MLP"
    )


def test_grow_layer_with_forageact():
    batch_size, in_dim, hidden_dim, out_dim = 2, 5, 8, 3
    X = np.random.randn(batch_size, in_dim).astype(np.float64)
    
    model2 = Sequential(
        Dense(in_dim, hidden_dim),
        ForageAct(mode="per_neuron", init_alpha=0.2, n_neurons=hidden_dim),
        Dense(hidden_dim, out_dim)
    )
    
    orig_output = model2.forward(X)
    
    # Grow by 4 neurons
    grow_layer(model2, layer_idx=0, num_new_neurons=4)
    
    # Check alpha dimension
    assert model2.layers[1].n_neurons == hidden_dim + 4
    assert model2.layers[1]._alpha_param.data.shape == (hidden_dim + 4,)
    
    new_output = model2.forward(X)
    np.testing.assert_allclose(
        orig_output, new_output, 
        rtol=1e-6, atol=1e-8,
        err_msg="Net2WiderNet did not preserve output for per-neuron ForageAct"
    )


def test_prune_layer():
    batch_size, in_dim, hidden_dim, out_dim = 2, 5, 10, 3
    X = np.random.randn(batch_size, in_dim).astype(np.float64)
    
    model = Sequential(
        Dense(in_dim, hidden_dim),
        ForageAct(mode="per_neuron", init_alpha=0.2, n_neurons=hidden_dim),
        Dense(hidden_dim, out_dim)
    )
    
    # Forward and backward to populate gradients (needed for structured pruning)
    out = model.forward(X)
    d_out = np.ones_like(out)
    model.backward(d_out)
    
    # Prune 20% of neurons -> 2 neurons should be removed, leaving 8
    prune_layer(model, layer_idx=0, prune_fraction=0.2)
    
    # Verify dimensions
    assert model.layers[0].output_dim == 8
    assert model.layers[0].W.data.shape == (in_dim, 8)
    assert model.layers[0].W.grad.shape == (in_dim, 8)
    
    assert model.layers[1].n_neurons == 8
    assert model.layers[1]._alpha_param.data.shape == (8,)
    
    assert model.layers[2].input_dim == 8
    assert model.layers[2].W.data.shape == (8, out_dim)
    
    # Ensure forward and backward still run without crashing
    out_pruned = model.forward(X)
    model.backward(np.ones_like(out_pruned))
    
    assert out_pruned.shape == (batch_size, out_dim)
