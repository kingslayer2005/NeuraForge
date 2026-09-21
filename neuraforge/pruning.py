"""
pruning.py — Structured gradient-based pruning.

Prunes the bottom p% of neurons in a layer based on their L2 weight norm
and outgoing gradient magnitude (Taylor approximation).
"""

import numpy as np

from neuraforge.model import Sequential
from neuraforge.layers import Dense


def prune_layer(model: Sequential, layer_idx: int, prune_fraction: float = 0.1) -> None:
    """Prune the least important neurons from a Dense layer.
    
    Importance is measured as the L2 norm of the outgoing weights 
    multiplied by their accumulated gradients.
    
    Parameters
    ----------
    model : Sequential
        The neural network.
    layer_idx : int
        The index of the Dense layer to prune.
    prune_fraction : float
        Fraction of neurons to remove (e.g. 0.1 for 10%).
    """
    if layer_idx >= len(model.layers) - 1:
        raise ValueError("Cannot prune the output layer.")
        
    layer = model.layers[layer_idx]
    if not isinstance(layer, Dense):
        raise ValueError("Can only prune Dense layers.")
        
    # Find the next Dense layer to remove the incoming connections
    next_dense_idx = -1
    for i in range(layer_idx + 1, len(model.layers)):
        if isinstance(model.layers[i], Dense):
            next_dense_idx = i
            break
            
    if next_dense_idx == -1:
        raise ValueError("No subsequent Dense layer found to adjust weights.")
        
    next_layer = model.layers[next_dense_idx]
    
    out_dim = layer.output_dim
    n_prune = int(out_dim * prune_fraction)
    if n_prune == 0:
        return
        
    if out_dim - n_prune <= 0:
        raise ValueError("Cannot prune all neurons in a layer.")
        
    # Compute importance of each neuron based on OUTGOING weights in the next layer
    # Score = || W_out * dL/dW_out ||_2
    
    # next_layer.W.data has shape (out_dim, next_layer.output_dim)
    # next_layer.W.grad should ideally be populated from a recent backward pass.
    # If gradients are zero/none, we fall back to pure magnitude pruning.
    grad = next_layer.W.grad if next_layer.W.grad is not None else np.ones_like(next_layer.W.data)
    
    # Score for each of the `out_dim` neurons (axis 1 of layer.W, axis 0 of next_layer.W)
    scores = np.linalg.norm(next_layer.W.data * grad, axis=1)
    
    # Get indices of neurons to keep (top scores)
    keep_indices = np.argsort(scores)[n_prune:]
    keep_indices = np.sort(keep_indices) # Keep original order for stability
    
    new_out_dim = out_dim - n_prune
    
    # 1. Update the target layer (remove columns of W, elements of b)
    layer.W.data = layer.W.data[:, keep_indices]
    if layer.W.grad is not None:
        layer.W.grad = layer.W.grad[:, keep_indices]
        
    layer.b.data = layer.b.data[:, keep_indices]
    if layer.b.grad is not None:
        layer.b.grad = layer.b.grad[:, keep_indices]
        
    layer.output_dim = new_out_dim
    
    # 2. Update the next layer (remove rows of W)
    next_layer.W.data = next_layer.W.data[keep_indices, :]
    if next_layer.W.grad is not None:
        next_layer.W.grad = next_layer.W.grad[keep_indices, :]
        
    next_layer.input_dim = new_out_dim
    
    # 3. Handle intermediate activations (e.g., ForageAct per_neuron)
    for i in range(layer_idx + 1, next_dense_idx):
        act = model.layers[i]
        if hasattr(act, "mode") and act.mode == "per_neuron":
            act.n_neurons = new_out_dim
            act._alpha_param.data = act._alpha_param.data[keep_indices]
            act._alpha_value = act._alpha_param.data
            if act._alpha_param.grad is not None:
                act._alpha_param.grad = act._alpha_param.grad[keep_indices]
