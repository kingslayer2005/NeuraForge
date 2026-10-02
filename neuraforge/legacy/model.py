"""
model.py — Sequential model container for NeuraForge.

Chains together layers and activations into a feed-forward network.
Crucially, manages the parameter registry so optimizers can locate and update
all learnable parameters across the entire network without hardcoded names.
"""

from __future__ import annotations

from typing import List, Any

import numpy as np

from neuraforge.layers import Parameter


class Sequential:
    """A sequential container for neural network layers.

    Layers are added in the order they should be executed.
    During forward propagation, the output of layer i is fed as input
    to layer i+1.

    Parameters
    ----------
    *layers : Any
        A variable number of layer objects (Dense, ReLU, Dropout, etc.).
        Every layer must implement:
            - forward(X, training=True) -> np.ndarray
            - backward(d_out) -> np.ndarray
            - parameters() -> list[Parameter]
    """

    def __init__(self, *layers: Any) -> None:
        # Store layers as a list (tuples are immutable; list allows dynamic changes later)
        self.layers: List[Any] = list(layers)

    @property
    def config(self) -> dict:
        """Return a dict describing the entire architecture."""
        return {
            "type": "Sequential",
            "layers": [layer.config for layer in self.layers if hasattr(layer, "config")],
        }

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Propagate inputs forward through all layers.

        Parameters
        ----------
        X : np.ndarray
            Input data batch.
        training : bool
            Whether the model is training (True) or evaluating (False).
            Passed to layers like Dropout that behave differently during eval.

        Returns
        -------
        np.ndarray
            Final layer output (e.g. logits).
        """
        out = X
        for layer in self.layers:
            # Pass output of previous layer as input to the next
            out = layer.forward(out, training=training)
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Propagate gradients backward through all layers.

        Parameters
        ----------
        d_out : np.ndarray
            Gradient of the loss w.r.t. the final layer's output.

        Returns
        -------
        np.ndarray
            Gradient w.r.t. the initial input X (often unused, but
            useful for input-gradient attribution methods).
        """
        # Traverse layers in reverse order for backpropagation
        d_prev = d_out
        for layer in reversed(self.layers):
            # Pass gradient from the next layer down to the current one
            d_prev = layer.backward(d_prev)
        return d_prev

    def parameters(self) -> List[Parameter]:
        """Collect all learnable parameters from all layers.

        This acts as a central registry. It walks through every layer,
        asks for its parameters, and renames them to include the layer
        index (e.g. "layer_0/W") to ensure global uniqueness.

        Returns
        -------
        list[Parameter]
            A flat list of Parameter objects ready for an optimizer.
        """
        all_params: List[Parameter] = []

        for i, layer in enumerate(self.layers):
            layer_params = layer.parameters()
            for p in layer_params:
                # Prefix the parameter name with the layer index
                # Example: "W" -> "layer_0/W"
                # Strip any existing prefix in case parameters() is called multiple times
                base_name = p.name.split("/")[-1]
                p.name = f"layer_{i}/{base_name}"
                all_params.append(p)

        return all_params

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run forward pass in evaluation mode.

        Parameters
        ----------
        X : np.ndarray
            Input data batch.

        Returns
        -------
        np.ndarray
            Final layer output (e.g. logits). Callers must apply argmax
            if they want discrete class predictions.
        """
        return self.forward(X, training=False)
