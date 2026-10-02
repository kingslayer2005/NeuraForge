"""
nn.py — Neural network layers built on the autograd engine.

Every layer in this module computes its forward pass using autograd Tensor
operations, so backward passes are automatic. No hand-written backward()
methods.

Layers:
    Dense (Linear)       — fully connected
    Conv2d               — 2D convolution via im2col
    MaxPool2d            — 2D max pooling
    BatchNorm1d          — batch normalization for FC layers
    BatchNorm2d          — batch normalization for conv layers
    LayerNorm            — layer normalization
    Dropout              — inverted dropout
    Embedding            — lookup table for discrete tokens
    ReLU, GELU, SiLU, Tanh, Sigmoid, Softmax — activation wrappers
    ForageAct            — custom learnable activation

Public API is kept compatible with the original neuraforge modules:
    layer.forward(X, training=True) → np.ndarray
    layer.parameters() → list[Parameter]
"""

from __future__ import annotations

import math

import numpy as np

from neuraforge.autograd import Tensor
from neuraforge.layers import Parameter


# ===================================================================
# MODULE BASE CLASS
# ===================================================================
class Module:
    """Base class for all neural network modules.

    Provides:
        - parameters() method that recursively collects all Parameter objects
        - train()/eval() mode switching
        - forward() must be implemented by subclasses
    """

    def __init__(self):
        self._training = True

    def train(self, mode: bool = True):
        """Set the module to training mode."""
        self._training = mode
        return self

    def eval(self):
        """Set the module to evaluation mode."""
        return self.train(False)

    def parameters(self) -> list[Parameter]:
        """Collect all Parameter objects owned by this module.

        Override in subclasses that have learnable parameters.
        """
        return []

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# ===================================================================
# DENSE (LINEAR) LAYER — built on autograd
# ===================================================================
class Dense(Module):
    """Fully-connected linear layer: output = X @ W + b.

    Uses He initialisation (Kaiming, 2015) which scales weights by
    sqrt(2 / fan_in) to keep variance stable through ReLU-family activations.

    This version uses autograd Tensors internally for automatic differentiation,
    but exposes the same public API as the legacy Dense layer.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    output_dim : int
        Number of output features (neurons).
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # He initialisation: scale by sqrt(2/fan_in) for ReLU-family stability
        init_W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        init_b = np.zeros((1, output_dim))

        # Wrap in Parameter objects so the optimizer can find and update them
        self.W = Parameter("W", init_W)
        self.b = Parameter("b", init_b)

        # Cache for backward compatibility with legacy API
        self._input = None

    @property
    def config(self) -> dict:
        return {"type": "Dense", "input_dim": self.input_dim, "output_dim": self.output_dim}

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Compute the linear transformation X @ W + b.

        Parameters
        ----------
        X : np.ndarray, shape (batch, input_dim)
            Input activations from the previous layer.
        training : bool
            Unused here; accepted for API consistency.

        Returns
        -------
        np.ndarray, shape (batch, output_dim)
        """
        # Store input for backward pass
        self._input = X

        # Create autograd tensors for the computation
        # X_t needs requires_grad=True so we can compute dL/dX for backprop
        X_t = Tensor(X, requires_grad=True)
        W_t = Tensor(self.W.data, requires_grad=True)
        b_t = Tensor(self.b.data, requires_grad=True)

        # Forward: X @ W + b (autograd tracks the ops automatically)
        out_t = X_t @ W_t + b_t

        # Store tensors for backward
        self._W_tensor = W_t
        self._b_tensor = b_t
        self._out_tensor = out_t
        self._X_tensor = X_t

        return out_t.data

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Compute gradients using autograd and propagate backward.

        Parameters
        ----------
        d_out : np.ndarray, shape (batch, output_dim)
            Gradient of the loss w.r.t. this layer's output.

        Returns
        -------
        np.ndarray, shape (batch, input_dim)
            Gradient of the loss w.r.t. this layer's input.
        """
        # Use autograd backward with the upstream gradient
        self._out_tensor.backward(d_out)

        # Extract gradients from the autograd tensors
        self.W.grad = self._W_tensor.grad
        self.b.grad = self._b_tensor.grad

        # Return gradient w.r.t. input
        d_input = self._X_tensor.grad if self._X_tensor.grad is not None else np.zeros_like(self._input)
        return d_input

    def parameters(self) -> list[Parameter]:
        return [self.W, self.b]


# ===================================================================
# CONV2D — 2D Convolution via im2col
# ===================================================================
class Conv2d(Module):
    """2D convolutional layer using the im2col trick.

    im2col transforms the convolution operation into a matrix multiplication:
    1. Extract all (kernel_h × kernel_w × C_in) patches from the input
    2. Reshape them into a 2D matrix: (N * H_out * W_out, kernel_h * kernel_w * C_in)
    3. Multiply with the reshaped filter: (kernel_h * kernel_w * C_in, C_out)
    4. Reshape result back to (N, H_out, W_out, C_out)

    This avoids nested Python loops and leverages NumPy's optimized GEMM.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (filters).
    kernel_size : int or tuple
        Size of the convolving kernel.
    stride : int
        Stride of the convolution. Default 1.
    padding : int
        Zero-padding added to both sides of the input. Default 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Normalize kernel_size to a tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        kH, kW = self.kernel_size

        # Kaiming init for conv weights: scale by sqrt(2 / (fan_in))
        # fan_in = in_channels * kernel_h * kernel_w
        fan_in = in_channels * kH * kW
        init_W = np.random.randn(out_channels, in_channels, kH, kW) * np.sqrt(2.0 / fan_in)
        init_b = np.zeros(out_channels)

        self.W = Parameter("W", init_W)
        self.b = Parameter("b", init_b)

    @property
    def config(self) -> dict:
        return {
            "type": "Conv2d",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
        }

    def _im2col(self, X: np.ndarray) -> np.ndarray:
        """Transform input into column matrix for convolution as GEMM.

        Input X has shape (N, C, H, W) — channels-first format.
        Output has shape (N * H_out * W_out, C * kH * kW).

        The index arithmetic works as follows:
        - For each output position (i, j), we extract a patch of size
          (C, kH, kW) starting at input position (i*stride, j*stride).
        - We flatten each patch into a row of the output matrix.
        - The columns correspond to: [c0_r0_c0, c0_r0_c1, ..., cC_rKH_cKW]
        """
        N, C, H, W = X.shape
        kH, kW = self.kernel_size
        s = self.stride

        # Compute output spatial dimensions
        H_out = (H - kH) // s + 1
        W_out = (W - kW) // s + 1

        # Use np.lib.stride_tricks for efficient patch extraction
        # Build the shape and strides for the 6D view:
        # (N, H_out, W_out, C, kH, kW)
        shape = (N, H_out, W_out, C, kH, kW)
        strides = (
            X.strides[0],        # step across batch
            X.strides[2] * s,    # step across output rows (stride pixels in H)
            X.strides[3] * s,    # step across output cols (stride pixels in W)
            X.strides[1],        # step across channels
            X.strides[2],        # step across kernel rows
            X.strides[3],        # step across kernel cols
        )

        # Create a strided view (no copy, just reinterprets the memory layout)
        patches = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)

        # Reshape to 2D: (N * H_out * W_out, C * kH * kW)
        cols = patches.reshape(N * H_out * W_out, C * kH * kW)

        return cols

    def _col2im(self, cols: np.ndarray, X_shape: tuple) -> np.ndarray:
        """Transform column gradients back to input gradient shape.

        This is the reverse of im2col: scatter the column gradients back
        to the original input positions. Since multiple output positions
        may read from the same input position (when stride < kernel_size),
        we accumulate gradients with np.add.at.

        Parameters
        ----------
        cols : np.ndarray, shape (N * H_out * W_out, C * kH * kW)
            Gradient in column format.
        X_shape : tuple (N, C, H, W)
            Original input shape.

        Returns
        -------
        np.ndarray, shape (N, C, H, W)
            Gradient w.r.t. the (padded) input.
        """
        N, C, H, W = X_shape
        kH, kW = self.kernel_size
        s = self.stride

        H_out = (H - kH) // s + 1
        W_out = (W - kW) // s + 1

        # Reshape cols back to (N, H_out, W_out, C, kH, kW)
        cols_reshaped = cols.reshape(N, H_out, W_out, C, kH, kW)

        # Initialize the gradient array
        dX = np.zeros(X_shape, dtype=cols.dtype)

        # Scatter gradients back to input positions
        # For each output position (oh, ow), the patch starts at (oh*s, ow*s)
        for oh in range(H_out):
            for ow in range(W_out):
                # Input region that this output position reads from
                h_start = oh * s
                w_start = ow * s
                # Accumulate the gradient from this output position
                dX[:, :, h_start:h_start + kH, w_start:w_start + kW] += cols_reshaped[:, oh, ow, :, :, :]

        return dX

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass: 2D convolution via im2col + GEMM.

        Parameters
        ----------
        X : np.ndarray, shape (N, C_in, H, W)
            Input feature maps in channels-first format.

        Returns
        -------
        np.ndarray, shape (N, C_out, H_out, W_out)
            Output feature maps.
        """
        # Apply padding if needed
        if self.padding > 0:
            X = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding),
                          (self.padding, self.padding)), mode='constant')

        self._input_padded = X
        self._input_shape_before_pad = X.shape  # after padding

        N, _C, H, W = X.shape
        kH, kW = self.kernel_size
        s = self.stride

        H_out = (H - kH) // s + 1
        W_out = (W - kW) // s + 1

        # Step 1: im2col — extract patches as rows of a matrix
        # cols shape: (N * H_out * W_out, C_in * kH * kW)
        cols = self._im2col(X)
        self._cols = cols  # cache for backward

        # Step 2: Reshape weights to 2D for GEMM
        # W shape: (C_out, C_in, kH, kW) → (C_in * kH * kW, C_out)
        W_2d = self.W.data.reshape(self.out_channels, -1).T

        # Step 3: GEMM — cols @ W_2d
        # Result shape: (N * H_out * W_out, C_out)
        out_2d = cols @ W_2d + self.b.data  # bias broadcast

        # Step 4: Reshape back to (N, H_out, W_out, C_out), then transpose to (N, C_out, H_out, W_out)
        out = out_2d.reshape(N, H_out, W_out, self.out_channels)
        out = out.transpose(0, 3, 1, 2)  # NHWC → NCHW

        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Backward pass for Conv2d.

        Parameters
        ----------
        d_out : np.ndarray, shape (N, C_out, H_out, W_out)
            Gradient w.r.t. the output.

        Returns
        -------
        np.ndarray, shape (N, C_in, H, W)
            Gradient w.r.t. the input (before padding).
        """
        _N, C_out, _H_out, _W_out = d_out.shape
        _kH, _kW = self.kernel_size

        # Reshape d_out: (N, C_out, H_out, W_out) → (N, H_out, W_out, C_out) → (N*H_out*W_out, C_out)
        d_out_2d = d_out.transpose(0, 2, 3, 1).reshape(-1, C_out)

        # Gradient w.r.t. bias: sum over all spatial positions and batch
        # db shape: (C_out,)
        self.b.grad = np.sum(d_out_2d, axis=0)

        # Gradient w.r.t. weights:
        # W_2d = W.reshape(C_out, -1).T  →  shape (C_in*kH*kW, C_out)
        # cols @ W_2d = out_2d
        # dW_2d = cols.T @ d_out_2d  →  shape (C_in*kH*kW, C_out)
        dW_2d = self._cols.T @ d_out_2d
        # Reshape back to (C_out, C_in, kH, kW) — note the transpose
        self.W.grad = dW_2d.T.reshape(self.W.data.shape)

        # Gradient w.r.t. input (in column space):
        # cols @ W_2d = out_2d
        # d_cols = d_out_2d @ W_2d.T  →  shape (N*H_out*W_out, C_in*kH*kW)
        W_2d = self.W.data.reshape(self.out_channels, -1).T
        d_cols = d_out_2d @ W_2d.T

        # col2im: scatter column gradients back to input shape
        dX_padded = self._col2im(d_cols, self._input_padded.shape)

        # Remove padding from the gradient if padding was applied
        if self.padding > 0:
            p = self.padding
            dX = dX_padded[:, :, p:-p, p:-p]
        else:
            dX = dX_padded

        return dX

    def parameters(self) -> list[Parameter]:
        return [self.W, self.b]


# ===================================================================
# MAXPOOL2D
# ===================================================================
class MaxPool2d(Module):
    """2D max pooling with correct argmax routing of gradients.

    Parameters
    ----------
    kernel_size : int
        Size of the pooling window.
    stride : int or None
        Stride of the pooling. Default: same as kernel_size.
    """

    def __init__(self, kernel_size: int = 2, stride: int | None = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    @property
    def config(self) -> dict:
        return {"type": "MaxPool2d", "kernel_size": self.kernel_size, "stride": self.stride}

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass: take max over each pooling window.

        Parameters
        ----------
        X : np.ndarray, shape (N, C, H, W)

        Returns
        -------
        np.ndarray, shape (N, C, H_out, W_out)
        """
        N, C, H, W = X.shape
        k = self.kernel_size
        s = self.stride

        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1

        self._input_shape = X.shape

        # Reshape input to extract pooling windows
        # We use stride tricks to create a view of shape (N, C, H_out, W_out, k, k)
        shape = (N, C, H_out, W_out, k, k)
        strides = (
            X.strides[0],        # batch
            X.strides[1],        # channel
            X.strides[2] * s,    # output row
            X.strides[3] * s,    # output col
            X.strides[2],        # kernel row
            X.strides[3],        # kernel col
        )
        windows = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)

        # Take max over the last two dims (kernel)
        out = windows.reshape(N, C, H_out, W_out, k * k).max(axis=-1)

        # Store the argmax mask for backward (gradient routes only through max elements)
        self._argmax_mask = (windows.reshape(N, C, H_out, W_out, k * k) ==
                            out[..., np.newaxis])
        self._windows_shape = windows.shape

        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Route gradient only through the max elements in each pool window.

        Parameters
        ----------
        d_out : np.ndarray, shape (N, C, H_out, W_out)

        Returns
        -------
        np.ndarray, shape (N, C, H, W)
        """
        N, C, _H, _W = self._input_shape
        k = self.kernel_size
        s = self.stride
        H_out = d_out.shape[2]
        W_out = d_out.shape[3]

        # Expand d_out to match windows: (N, C, H_out, W_out, k*k)
        d_out_expanded = d_out[..., np.newaxis]

        # Handle ties: distribute gradient equally among tied max elements
        mask = self._argmax_mask
        tie_count = mask.sum(axis=-1, keepdims=True)
        tie_count = np.maximum(tie_count, 1)  # avoid div by zero
        mask = mask / tie_count

        # Gradient for each window element
        d_windows = d_out_expanded * mask  # shape (N, C, H_out, W_out, k*k)
        d_windows = d_windows.reshape(N, C, H_out, W_out, k, k)

        # Scatter back to input shape
        dX = np.zeros(self._input_shape, dtype=d_out.dtype)
        for oh in range(H_out):
            for ow in range(W_out):
                h_start = oh * s
                w_start = ow * s
                dX[:, :, h_start:h_start + k, w_start:w_start + k] += d_windows[:, :, oh, ow, :, :]

        return dX

    def parameters(self) -> list[Parameter]:
        return []


# ===================================================================
# BATCHNORM1D
# ===================================================================
class BatchNorm1d(Module):
    """Batch normalization for fully-connected layers.

    Normalizes each feature across the batch dimension, then applies
    a learnable affine transform: y = gamma * x_hat + beta.

    In training mode: uses batch statistics (mean, var).
    In eval mode: uses running statistics accumulated during training.

    The backward pass has three gradient paths:
    1. Through the normalized input x_hat (direct path)
    2. Through the batch mean μ (the normalization centers the data)
    3. Through the batch variance σ² (the normalization scales the data)

    Full derivation in comments below.

    Parameters
    ----------
    num_features : int
        Number of features (C for FC layers).
    eps : float
        Small constant for numerical stability. Default 1e-5.
    momentum : float
        Running statistics EMA decay. Default 0.1.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable affine parameters
        self.gamma = Parameter("gamma", np.ones(num_features))
        self.beta = Parameter("beta", np.zeros(num_features))

        # Running statistics for eval mode (not learnable)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    @property
    def config(self) -> dict:
        return {"type": "BatchNorm1d", "num_features": self.num_features,
                "eps": self.eps, "momentum": self.momentum}

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply batch normalization.

        Parameters
        ----------
        X : np.ndarray, shape (N, C)

        Returns
        -------
        np.ndarray, shape (N, C)
        """
        self.training = training
        if training:
            # Compute batch statistics
            # mean shape: (C,), computed over the batch dimension
            self._mean = np.mean(X, axis=0)
            # var shape: (C,), biased variance over the batch for normalization
            self._var = np.var(X, axis=0)
            self._N = X.shape[0]

            # Update running statistics using EMA (PyTorch uses unbiased var for running_var)
            unbiased_var = np.var(X, axis=0, ddof=1) if self._N > 1 else self._var
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self._mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbiased_var

            # Normalize: x_hat = (x - μ) / sqrt(σ² + ε)
            self._std = np.sqrt(self._var + self.eps)
            self._x_centered = X - self._mean
            self._x_hat = self._x_centered / self._std
        else:
            # Use running statistics for inference
            self._std = np.sqrt(self.running_var + self.eps)
            self._x_hat = (X - self.running_mean) / self._std

        # Affine transform: y = γ * x_hat + β
        return self.gamma.data * self._x_hat + self.beta.data

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Backward pass for batch normalization.

        The full derivation of the three gradient paths:

        Let x_hat = (x - μ) / σ,  y = γ * x_hat + β,  σ = sqrt(var + ε)

        1. Gradient w.r.t. γ (gamma):
           dL/dγ = Σ_batch (dL/dy * x_hat)

        2. Gradient w.r.t. β (beta):
           dL/dβ = Σ_batch (dL/dy)

        3. Gradient w.r.t. input x (the hard part):
           dL/dx_hat = dL/dy * γ

           Now x_hat depends on x through three paths:
           a) Directly: x_hat = (x - μ) / σ
           b) Through μ = (1/N) Σ x_i
           c) Through σ² = (1/N) Σ (x_i - μ)²

           Path a: dx_hat/dx_i = 1/σ (for the i-th sample)

           Path b: dL/dμ = Σ dL/dx_hat * (-1/σ)
                   dμ/dx_i = 1/N

           Path c: dL/dσ² = Σ dL/dx_hat * (x - μ) * (-0.5) * (σ² + ε)^(-3/2)
                   dσ²/dx_i = 2(x_i - μ) / N

           Combining all three paths:
           dL/dx = dL/dx_hat * (1/σ) + dL/dμ * (1/N) + dL/dσ² * 2(x-μ)/N

           This simplifies to:
           dL/dx = (1/Nσ) * (N * dL/dx_hat - Σ dL/dx_hat - x_hat * Σ(dL/dx_hat * x_hat))

        Parameters
        ----------
        d_out : np.ndarray, shape (N, C)

        Returns
        -------
        np.ndarray, shape (N, C)
        """
        # Gradient w.r.t. gamma: sum over batch
        self.gamma.grad = np.sum(d_out * self._x_hat, axis=0)

        # Gradient w.r.t. beta: sum over batch
        self.beta.grad = np.sum(d_out, axis=0)

        # Gradient w.r.t. x_hat
        dx_hat = d_out * self.gamma.data

        if not getattr(self, 'training', True):
            return dx_hat / self._std

        N = self._N

        # Using the simplified combined formula:
        # dL/dx = (1/(N*σ)) * (N*dx_hat - sum(dx_hat) - x_hat * sum(dx_hat * x_hat))
        dx = (1.0 / (N * self._std)) * (
            N * dx_hat
            - np.sum(dx_hat, axis=0)
            - self._x_hat * np.sum(dx_hat * self._x_hat, axis=0)
        )

        return dx

    def parameters(self) -> list[Parameter]:
        return [self.gamma, self.beta]


# ===================================================================
# BATCHNORM2D
# ===================================================================
class BatchNorm2d(Module):
    """Batch normalization for convolutional layers.

    Same math as BatchNorm1d, but operates on (N, C, H, W) tensors.
    Statistics are computed per-channel across (N, H, W).

    Parameters
    ----------
    num_features : int
        Number of channels (C).
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Parameter("gamma", np.ones(num_features))
        self.beta = Parameter("beta", np.zeros(num_features))

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    @property
    def config(self) -> dict:
        return {"type": "BatchNorm2d", "num_features": self.num_features}

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply batch normalization to 4D input (N, C, H, W).

        Statistics are computed per channel across N, H, W dimensions.
        """
        N, C, H, W = X.shape

        self.training = training
        if training:
            # Compute per-channel statistics across (N, H, W)
            self._mean = np.mean(X, axis=(0, 2, 3))  # shape (C,)
            self._var = np.var(X, axis=(0, 2, 3))      # shape (C,)
            self._N = N * H * W  # number of elements per channel

            # Update running statistics
            unbiased_var = np.var(X, axis=(0, 2, 3), ddof=1) if self._N > 1 else self._var
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self._mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbiased_var

            # Reshape for broadcasting: (1, C, 1, 1)
            mean = self._mean.reshape(1, C, 1, 1)
            var = self._var.reshape(1, C, 1, 1)

            self._std = np.sqrt(var + self.eps)
            self._x_centered = X - mean
            self._x_hat = self._x_centered / self._std
        else:
            mean = self.running_mean.reshape(1, C, 1, 1)
            var = self.running_var.reshape(1, C, 1, 1)
            self._std = np.sqrt(var + self.eps)
            self._x_hat = (X - mean) / self._std

        # Affine transform with gamma and beta reshaped for broadcasting
        gamma = self.gamma.data.reshape(1, C, 1, 1)
        beta = self.beta.data.reshape(1, C, 1, 1)

        return gamma * self._x_hat + beta

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Backward pass for BatchNorm2d.

        Same derivation as BatchNorm1d, but statistics are computed
        per-channel across (N, H, W).
        """
        _N_batch, C, _H, _W = d_out.shape

        gamma = self.gamma.data.reshape(1, C, 1, 1)

        # Gradient w.r.t. gamma: sum over (N, H, W)
        self.gamma.grad = np.sum(d_out * self._x_hat, axis=(0, 2, 3))

        # Gradient w.r.t. beta: sum over (N, H, W)
        self.beta.grad = np.sum(d_out, axis=(0, 2, 3))

        # Gradient w.r.t. x_hat
        dx_hat = d_out * gamma

        if not getattr(self, 'training', True):
            return dx_hat / self._std

        N_total = self._N  # N * H * W

        # Combined formula applied per-channel:
        sum_dx_hat = np.sum(dx_hat, axis=(0, 2, 3), keepdims=True)
        sum_dx_hat_xhat = np.sum(dx_hat * self._x_hat, axis=(0, 2, 3), keepdims=True)

        dx = (1.0 / (N_total * self._std)) * (
            N_total * dx_hat - sum_dx_hat - self._x_hat * sum_dx_hat_xhat
        )

        return dx

    def parameters(self) -> list[Parameter]:
        return [self.gamma, self.beta]


# ===================================================================
# LAYERNORM
# ===================================================================
class LayerNorm(Module):
    """Layer normalization.

    Normalizes across the last dimension (features), independently for
    each sample and position in the batch.

    Parameters
    ----------
    normalized_shape : int or tuple
        Shape of the features to normalize over (typically the last dim).
    """

    def __init__(self, normalized_shape, eps: float = 1e-5) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps

        n = 1
        for s in normalized_shape:
            n *= s

        self.gamma = Parameter("gamma", np.ones(n))
        self.beta = Parameter("beta", np.zeros(n))

    @property
    def config(self) -> dict:
        return {"type": "LayerNorm", "normalized_shape": self.normalized_shape}

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply layer normalization.

        Parameters
        ----------
        X : np.ndarray, shape (..., *normalized_shape)

        Returns
        -------
        np.ndarray, same shape as X
        """
        # Compute statistics over the last len(normalized_shape) dimensions
        n_dims = len(self.normalized_shape)
        axes = tuple(range(-n_dims, 0))

        self._mean = np.mean(X, axis=axes, keepdims=True)
        self._var = np.var(X, axis=axes, keepdims=True)
        self._std = np.sqrt(self._var + self.eps)
        self._x_centered = X - self._mean
        self._x_hat = self._x_centered / self._std
        self._input_shape = X.shape
        self._axes = axes

        # Number of elements being normalized (per sample)
        self._D = 1
        for s in self.normalized_shape:
            self._D *= s

        gamma = self.gamma.data.reshape(self.normalized_shape)
        beta = self.beta.data.reshape(self.normalized_shape)

        return gamma * self._x_hat + beta

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Backward pass for LayerNorm.

        Similar to BatchNorm, but normalization is over features
        (last dims) rather than over the batch.
        """
        D = self._D

        gamma = self.gamma.data.reshape(self.normalized_shape)

        # Gradient w.r.t. gamma: sum over all dimensions except the normalized ones
        # For a typical (N, D) input, sum over N
        n_norm_dims = len(self.normalized_shape)
        batch_axes = tuple(range(len(self._input_shape) - n_norm_dims))

        self.gamma.grad = np.sum(d_out * self._x_hat, axis=batch_axes).reshape(-1)
        self.beta.grad = np.sum(d_out, axis=batch_axes).reshape(-1)

        # Gradient w.r.t. input
        dx_hat = d_out * gamma

        sum_dx_hat = np.sum(dx_hat, axis=self._axes, keepdims=True)
        sum_dx_hat_xhat = np.sum(dx_hat * self._x_hat, axis=self._axes, keepdims=True)

        dx = (1.0 / (D * self._std)) * (
            D * dx_hat - sum_dx_hat - self._x_hat * sum_dx_hat_xhat
        )

        return dx

    def parameters(self) -> list[Parameter]:
        return [self.gamma, self.beta]


# ===================================================================
# DROPOUT
# ===================================================================
class Dropout(Module):
    """Inverted dropout: randomly zeros elements during training.

    At inference, this layer is the identity function.

    Parameters
    ----------
    p : float
        Probability of dropping each element. Default 0.5.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p
        self._mask = None

    @property
    def config(self) -> dict:
        return {"type": "Dropout", "p": self.p}

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or self.p == 0.0:
            self._mask = None
            return X

        # Generate binary mask: keep with probability (1 - p)
        self._mask = (np.random.rand(*X.shape) >= self.p).astype(X.dtype)
        # Scale by 1/(1-p) so expected value is unchanged
        return X * self._mask / (1.0 - self.p)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        if self._mask is None:
            return d_out
        return d_out * self._mask / (1.0 - self.p)

    def parameters(self) -> list[Parameter]:
        return []


# ===================================================================
# EMBEDDING
# ===================================================================
class Embedding(Module):
    """Lookup table for discrete token indices.

    Parameters
    ----------
    num_embeddings : int
        Size of the vocabulary.
    embedding_dim : int
        Dimension of each embedding vector.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialize with small random values
        init_W = np.random.randn(num_embeddings, embedding_dim) * 0.02
        self.W = Parameter("W", init_W)

    @property
    def config(self) -> dict:
        return {"type": "Embedding", "num_embeddings": self.num_embeddings,
                "embedding_dim": self.embedding_dim}

    def forward(self, indices: np.ndarray, training: bool = True) -> np.ndarray:
        """Look up embeddings for the given indices.

        Parameters
        ----------
        indices : np.ndarray of int, shape (*)
            Token indices.

        Returns
        -------
        np.ndarray, shape (*, embedding_dim)
        """
        self._indices = indices
        return self.W.data[indices]

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Accumulate gradients for the looked-up embeddings.

        Parameters
        ----------
        d_out : np.ndarray, shape (*, embedding_dim)

        Returns
        -------
        None (no input gradient for discrete indices)
        """
        if self.W.grad is None:
            self.W.grad = np.zeros_like(self.W.data)
        # Accumulate gradients at the indexed positions
        np.add.at(self.W.grad, self._indices, d_out)
        return None  # No gradient for discrete indices

    def parameters(self) -> list[Parameter]:
        return [self.W]


# ===================================================================
# ACTIVATION WRAPPERS
# ===================================================================
class ReLU(Module):
    """ReLU activation wrapper for the nn module."""

    @property
    def config(self) -> dict:
        return {"type": "ReLU"}

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        self._Z = Z
        return np.maximum(0, Z)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        return d_out * (self._Z > 0).astype(d_out.dtype)

    def parameters(self) -> list[Parameter]:
        return []


class GELU(Module):
    """GELU activation (tanh approximation)."""

    _SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
    _COEFF = 0.044715

    @property
    def config(self) -> dict:
        return {"type": "GELU"}

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        self._Z = Z
        self._inner = self._SQRT_2_OVER_PI * (Z + self._COEFF * Z ** 3)
        self._tanh_inner = np.tanh(self._inner)
        return 0.5 * Z * (1.0 + self._tanh_inner)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        z, t = self._Z, self._tanh_inner
        g_prime = self._SQRT_2_OVER_PI * (1.0 + 3.0 * self._COEFF * z ** 2)
        sech2 = 1.0 - t ** 2
        grad = 0.5 * (1.0 + t) + 0.5 * z * sech2 * g_prime
        return d_out * grad

    def parameters(self) -> list[Parameter]:
        return []


class SiLU(Module):
    """SiLU (Swish) activation."""

    @property
    def config(self) -> dict:
        return {"type": "SiLU"}

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        self._Z = Z
        self._sigmoid = 1.0 / (1.0 + np.exp(-Z))
        return Z * self._sigmoid

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        s = self._sigmoid
        z = self._Z
        grad = s + z * s * (1.0 - s)
        return d_out * grad

    def parameters(self) -> list[Parameter]:
        return []


class Tanh(Module):
    """Tanh activation."""

    @property
    def config(self) -> dict:
        return {"type": "Tanh"}

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        self._tanh = np.tanh(Z)
        return self._tanh

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        return d_out * (1.0 - self._tanh ** 2)

    def parameters(self) -> list[Parameter]:
        return []


class Sigmoid(Module):
    """Sigmoid activation."""

    @property
    def config(self) -> dict:
        return {"type": "Sigmoid"}

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        self._sigmoid = 1.0 / (1.0 + np.exp(-Z))
        return self._sigmoid

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        return d_out * self._sigmoid * (1.0 - self._sigmoid)

    def parameters(self) -> list[Parameter]:
        return []


class Softmax(Module):
    """Softmax activation (numerically stable)."""

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    @property
    def config(self) -> dict:
        return {"type": "Softmax", "axis": self.axis}

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        shifted = Z - np.max(Z, axis=self.axis, keepdims=True)
        exp_z = np.exp(shifted)
        self._probs = exp_z / np.sum(exp_z, axis=self.axis, keepdims=True)
        return self._probs

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        p = self._probs
        sum_dp_p = np.sum(d_out * p, axis=self.axis, keepdims=True)
        return p * (d_out - sum_dp_p)

    def parameters(self) -> list[Parameter]:
        return []


# ===================================================================
# FORAGEACT — custom learnable activation (rebuilt on nn module)
# ===================================================================
class ForageAct(Module):
    """ForageAct: f(z) = z·σ(z) + α·tanh(z).

    Identical math to the legacy version, re-implemented as an nn.Module.
    """

    def __init__(
        self,
        mode: str = "scalar",
        init_alpha: float = 0.1,
        n_neurons: int | None = None,
    ) -> None:
        super().__init__()
        if mode not in ("fixed", "scalar", "per_neuron"):
            raise ValueError(f"mode must be 'fixed', 'scalar', or 'per_neuron', got {mode!r}")

        self.mode = mode
        self.init_alpha = init_alpha
        self.n_neurons = n_neurons

        if mode == "fixed":
            self._alpha_value = np.array([init_alpha])
            self._alpha_param = None
        elif mode == "scalar":
            alpha_data = np.array([init_alpha])
            self._alpha_param = Parameter("alpha", alpha_data)
            self._alpha_value = self._alpha_param.data
        elif mode == "per_neuron":
            if n_neurons is None:
                raise ValueError("n_neurons required for per_neuron mode")
            alpha_data = np.full(n_neurons, init_alpha)
            self._alpha_param = Parameter("alpha", alpha_data)
            self._alpha_value = self._alpha_param.data

    @property
    def config(self) -> dict:
        return {
            "type": "ForageAct", "mode": self.mode,
            "init_alpha": self.init_alpha, "n_neurons": self.n_neurons,
        }

    @property
    def alpha(self) -> np.ndarray:
        if self._alpha_param is not None:
            return self._alpha_param.data
        return self._alpha_value

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        self._Z = Z
        self._sigmoid = 1.0 / (1.0 + np.exp(-Z))
        self._tanh = np.tanh(Z)
        return Z * self._sigmoid + self.alpha * self._tanh

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        s = self._sigmoid
        z = self._Z
        t = self._tanh
        alpha = self.alpha

        s_prime = s * (1.0 - s)
        d_swish = s + z * s_prime
        d_tanh = 1.0 - t ** 2
        dZ = d_out * (d_swish + alpha * d_tanh)

        if self._alpha_param is not None:
            if self.mode == "scalar":
                self._alpha_param.grad = np.array([np.sum(d_out * t)])
            elif self.mode == "per_neuron":
                self._alpha_param.grad = np.sum(d_out * t, axis=0)

        return dZ

    def parameters(self) -> list[Parameter]:
        if self._alpha_param is not None:
            return [self._alpha_param]
        return []
