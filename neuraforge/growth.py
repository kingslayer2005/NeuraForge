"""
growth.py — Dynamic neuron growth (Net2WiderNet).

Implements function-preserving widening of a hidden layer as proposed
in Net2Net (Chen et al., 2015). When a layer is widened, the new neurons
are initialised such that the network outputs exactly the same values as before.
"""

import numpy as np

from neuraforge.layers import Dense
from neuraforge.model import Sequential


def grow_layer(model: Sequential, layer_idx: int, num_new_neurons: int) -> None:
    """Widen a hidden Dense layer while preserving the network function.

    Net2WiderNet algorithm:
    1. For the target layer (L), copy existing neurons to fill the new capacity.
       We add noise to the weights to break symmetry, so they can learn independently.
    2. For the next layer (L+1), scale the incoming weights from the copied neurons
       so that the sum of their contributions equals the original neuron's contribution.

    Parameters
    ----------
    model : Sequential
        The neural network.
    layer_idx : int
        The index of the Dense layer to widen.
    num_new_neurons : int
        Number of neurons to add to the layer.
    """
    if layer_idx >= len(model.layers) - 1:
        raise ValueError("Cannot widen the output layer.")
        
    layer = model.layers[layer_idx]
    if not isinstance(layer, Dense):
        raise TypeError("Can only widen Dense layers.")
        
    # Find the next Dense layer
    next_dense_idx = -1
    for i in range(layer_idx + 1, len(model.layers)):
        if isinstance(model.layers[i], Dense):
            next_dense_idx = i
            break
            
    if next_dense_idx == -1:
        raise ValueError("No subsequent Dense layer found to adjust weights.")
        
    next_layer = model.layers[next_dense_idx]
    
    # Original sizes
    in_dim = layer.input_dim
    old_out_dim = layer.output_dim
    new_out_dim = old_out_dim + num_new_neurons
    
    # 1. Select random existing neurons to replicate
    # Ensure we use the global numpy rng state
    replicate_indices = np.random.choice(old_out_dim, size=num_new_neurons, replace=True)
    
    # Track how many times each old neuron is represented in the NEW architecture
    # (Original count is 1, plus however many times we replicate it)
    replication_counts = np.ones(old_out_dim, dtype=int)
    for idx in replicate_indices:
        replication_counts[idx] += 1
        
    # 2. Build new weights and biases for the target layer
    W_new = np.zeros((in_dim, new_out_dim), dtype=layer.W.data.dtype)
    b_new = np.zeros((1, new_out_dim), dtype=layer.b.data.dtype)
    
    # Copy original neurons
    W_new[:, :old_out_dim] = layer.W.data
    b_new[:, :old_out_dim] = layer.b.data
    
    # Copy replicated neurons (exact copy, no noise)
    for i, orig_idx in enumerate(replicate_indices):
        new_idx = old_out_dim + i
        W_new[:, new_idx] = layer.W.data[:, orig_idx] # exact copy, no noise
        b_new[:, new_idx] = layer.b.data[:, orig_idx]
        
    # 3. Build new weights for the next layer
    W_next_new = np.zeros((new_out_dim, next_layer.output_dim), dtype=next_layer.W.data.dtype)
    
    # For every neuron (original and replicated), scale its outgoing weights by 1/count
    for i in range(old_out_dim):
        count = replication_counts[i]
        W_next_new[i, :] = next_layer.W.data[i, :] / count
        
    for i, orig_idx in enumerate(replicate_indices):
        new_idx = old_out_dim + i
        count = replication_counts[orig_idx]
        W_next_new[new_idx, :] = next_layer.W.data[orig_idx, :] / count
        
    # 4. Update the layers
    layer.output_dim = new_out_dim
    layer.W.data = W_new
    layer.b.data = b_new
    
    next_layer.input_dim = new_out_dim
    next_layer.W.data = W_next_new
    
    # Handle activations that depend on layer size (ForageAct per_neuron)
    # We must also widen the alpha parameter
    for i in range(layer_idx + 1, next_dense_idx):
        act = model.layers[i]
        if hasattr(act, "mode") and act.mode == "per_neuron":
            old_alpha = act.alpha
            new_alpha = np.zeros(new_out_dim, dtype=old_alpha.dtype)
            new_alpha[:old_out_dim] = old_alpha
            
            for j, orig_idx in enumerate(replicate_indices):
                new_idx = old_out_dim + j
                new_alpha[new_idx] = old_alpha[orig_idx]
                
            act.n_neurons = new_out_dim
            act._alpha_param.data = new_alpha
            act._alpha_value = act._alpha_param.data
