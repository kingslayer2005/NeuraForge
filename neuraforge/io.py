"""
io.py — Save and load model weights and architecture configurations.

Saves parameter values and layer configs as an uncompressed .npz file.
Allows for complete model reconstruction from the file alone.
"""

import json
from typing import Any, Dict

import numpy as np


def get_weights(model: Any) -> Dict[str, np.ndarray]:
    """Extract a dictionary of parameter copies from the model."""
    weights = {}
    for p in model.parameters():
        weights[p.name] = p.data.copy()
    return weights


def set_weights(model: Any, weights: Dict[str, np.ndarray]) -> None:
    """Load weights into the model from a dictionary in-place."""
    model_params = {p.name: p for p in model.parameters()}
    
    for name, data in weights.items():
        if name in model_params:
            if model_params[name].data.shape != data.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: model has {model_params[name].data.shape}, "
                    f"weights have {data.shape}"
                )
            # Copy data in-place so optimizers holding the Parameter object see the change
            model_params[name].data[:] = data
        else:
            raise KeyError(f"Weight {name} found in dict but not in model.")


def save_model(model: Any, filepath: str) -> None:
    """Save model architecture and weights to an .npz file.

    Parameters
    ----------
    model : Sequential
        The model to save.
    filepath : str
        Path to the output .npz file.
    """
    if not filepath.endswith(".npz"):
        filepath += ".npz"

    # Get the architecture config as a JSON string
    config_str = json.dumps(model.config)

    # Get the weights
    weights = get_weights(model)

    # Save to uncompressed npz. We store the config string as a 0D array.
    np.savez(filepath, __config__=np.array(config_str), **weights)


def build_from_config(config: dict) -> Any:
    """Reconstruct a model from its configuration dictionary."""
    from neuraforge.model import Sequential
    import neuraforge.layers as layers
    import neuraforge.activations as activations

    if config.get("type") != "Sequential":
        raise ValueError("Only Sequential models are supported for reconstruction.")

    model_layers = []
    for layer_cfg in config["layers"]:
        l_type = layer_cfg["type"]
        
        # Try to find the class in layers or activations
        if hasattr(layers, l_type):
            cls = getattr(layers, l_type)
        elif hasattr(activations, l_type):
            cls = getattr(activations, l_type)
        else:
            raise ValueError(f"Unknown layer type: {l_type}")

        # Extract arguments by removing 'type'
        kwargs = {k: v for k, v in layer_cfg.items() if k != "type"}
        model_layers.append(cls(**kwargs))

    return Sequential(*model_layers)


def load_model(filepath: str) -> Any:
    """Load a model's architecture and weights from an .npz file.

    Parameters
    ----------
    filepath : str
        Path to the .npz file.

    Returns
    -------
    Sequential
        The reconstructed model with loaded weights.
    """
    if not filepath.endswith(".npz"):
        filepath += ".npz"

    data = np.load(filepath, allow_pickle=True)
    
    if "__config__" not in data:
        raise ValueError(f"File {filepath} does not contain an architecture configuration.")

    # Reconstruct the model
    config_str = str(data["__config__"].item())
    config = json.loads(config_str)
    model = build_from_config(config)

    # Load the weights
    weights = {k: v for k, v in data.items() if k != "__config__"}
    set_weights(model, weights)

    return model
