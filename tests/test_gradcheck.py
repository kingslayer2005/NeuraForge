"""
test_gradcheck.py — Finite difference gradient checks for NeuraForge.

Verifies that the analytical gradients implemented in backward() match
the numerical gradients (finite differences) to within 1e-6 relative error.
All checks are performed in float64 for maximum precision.
"""

import numpy as np
import pytest

from neuraforge.activations import GELU, ForageAct, LeakyReLU, ReLU, Sigmoid, SiLU, Tanh
from neuraforge.layers import Dense
from neuraforge.losses import BCEWithLogitsLoss, MSELoss, SoftmaxCrossEntropy
from neuraforge.seed import seed_everything


def check_gradients(
    func,
    x,
    analytical_grad,
    eps=1e-5,
    rtol=1e-6,
    atol=1e-8,
):
    """
    Compare analytical gradients against central finite differences.
    
    func: a callable that takes a single array `x` and returns a scalar loss.
    x: the input array at which to evaluate the gradient.
    analytical_grad: the analytically computed gradient of func at x (same shape as x).
    """
    # Create a copy so we don't mutate the original array
    x_copy = x.copy()
    x_flat = x_copy.reshape(-1)
    numerical_grad_flat = np.zeros_like(x_flat)
    
    for i in range(len(x_flat)):
        orig_val = x_flat[i]
        
        # f(x + eps)
        x_flat[i] = orig_val + eps
        loss_plus = func(x_copy.reshape(x.shape))
        
        # f(x - eps)
        x_flat[i] = orig_val - eps
        loss_minus = func(x_copy.reshape(x.shape))
        
        # Restore original value
        x_flat[i] = orig_val
        
        # Central difference: (f(x+eps) - f(x-eps)) / (2*eps)
        numerical_grad_flat[i] = (loss_plus - loss_minus) / (2.0 * eps)
        
    numerical_grad = numerical_grad_flat.reshape(x.shape)
    
    np.testing.assert_allclose(
        analytical_grad,
        numerical_grad,
        rtol=rtol,
        atol=atol,
        err_msg="Analytical and numerical gradients mismatch."
    )


@pytest.fixture(autouse=True)
def setup_seed():
    seed_everything(42)


# --- Layer Gradient Checks ---

def test_dense_gradients():
    batch_size, in_dim, out_dim = 4, 5, 3
    X = np.random.randn(batch_size, in_dim).astype(np.float64)
    # Random scalar "loss" gradient from the layer above
    d_out = np.random.randn(batch_size, out_dim).astype(np.float64)
    
    layer = Dense(in_dim, out_dim)
    layer.W.data = layer.W.data.astype(np.float64)
    layer.b.data = layer.b.data.astype(np.float64)
    
    # 1. Forward and analytical backward
    _ = layer.forward(X)
    d_input = layer.backward(d_out)
    dW = layer.W.grad
    db = layer.b.grad
    
    # We define the scalar "loss" as sum(out * d_out)
    # so that the gradient of this loss w.r.t 'out' is exactly 'd_out'.
    
    # 2. Numerical check for input X
    def func_X(x_val):
        # We need a fresh forward pass because layer mutates internal state
        return np.sum((x_val @ layer.W.data + layer.b.data) * d_out)
        
    check_gradients(func_X, X, d_input)
    
    # 3. Numerical check for W
    def func_W(w_val):
        return np.sum((X @ w_val + layer.b.data) * d_out)
        
    check_gradients(func_W, layer.W.data, dW)
    
    # 4. Numerical check for b
    def func_b(b_val):
        return np.sum((X @ layer.W.data + b_val) * d_out)
        
    check_gradients(func_b, layer.b.data, db)


# --- Activation Gradient Checks ---

@pytest.mark.parametrize("ActClass, kwargs", [
    (ReLU, {}),
    (LeakyReLU, {"negative_slope": 0.01}),
    (Tanh, {}),
    (Sigmoid, {}),
    (SiLU, {}),
    (GELU, {}),
    (ForageAct, {"mode": "fixed", "init_alpha": 0.5}),
])
def test_activation_gradients(ActClass, kwargs):
    X = np.random.randn(5, 4).astype(np.float64)
    # Ensure no values are too close to exactly 0 for ReLU/LeakyReLU to avoid non-differentiable points
    X[np.abs(X) < 1e-4] = 0.1
    
    d_out = np.random.randn(5, 4).astype(np.float64)
    
    act = ActClass(**kwargs)
    
    act.forward(X)
    d_input = act.backward(d_out)
    
    def func_X(x_val):
        # Fresh forward pass
        return np.sum(act.forward(x_val) * d_out)
        
    check_gradients(func_X, X, d_input)


# --- ForageAct Alpha Gradient Checks ---

@pytest.mark.parametrize("mode", ["scalar", "per_neuron"])
def test_forageact_alpha_gradients(mode):
    batch_size, n_neurons = 5, 4
    X = np.random.randn(batch_size, n_neurons).astype(np.float64)
    d_out = np.random.randn(batch_size, n_neurons).astype(np.float64)
    
    act = ForageAct(mode=mode, init_alpha=0.5, n_neurons=n_neurons)
    act.parameters()[0].data = act.parameters()[0].data.astype(np.float64)
    
    # Forward and analytical backward
    act.forward(X)
    act.backward(d_out)
    d_alpha = act.parameters()[0].grad
    
    # Numerical check for alpha
    def func_alpha(alpha_val):
        # Temporarily overwrite alpha, compute scalar loss
        orig_alpha = act.parameters()[0].data.copy()
        act.parameters()[0].data[:] = alpha_val
        val = np.sum(act.forward(X) * d_out)
        act.parameters()[0].data[:] = orig_alpha
        return val
        
    check_gradients(func_alpha, act.parameters()[0].data, d_alpha)


# --- Loss Function Gradient Checks ---

def test_softmax_crossentropy_gradients():
    batch_size, num_classes = 5, 3
    logits = np.random.randn(batch_size, num_classes).astype(np.float64)
    # Random one-hot targets
    targets = np.eye(num_classes)[np.random.choice(num_classes, batch_size)].astype(np.float64)
    
    loss_fn = SoftmaxCrossEntropy()
    loss_fn.forward(logits, targets)
    d_logits = loss_fn.backward()
    
    def func_logits(log_val):
        return loss_fn.forward(log_val, targets)
        
    check_gradients(func_logits, logits, d_logits)


def test_mse_gradients():
    batch_size, dim = 5, 4
    preds = np.random.randn(batch_size, dim).astype(np.float64)
    targets = np.random.randn(batch_size, dim).astype(np.float64)
    
    loss_fn = MSELoss()
    loss_fn.forward(preds, targets)
    d_preds = loss_fn.backward()
    
    def func_preds(p_val):
        return loss_fn.forward(p_val, targets)
        
    check_gradients(func_preds, preds, d_preds)


def test_bcewithlogits_gradients():
    batch_size, dim = 5, 4
    logits = np.random.randn(batch_size, dim).astype(np.float64)
    targets = np.random.randint(0, 2, size=(batch_size, dim)).astype(np.float64)
    
    loss_fn = BCEWithLogitsLoss()
    loss_fn.forward(logits, targets)
    d_logits = loss_fn.backward()
    
    def func_logits(log_val):
        return loss_fn.forward(log_val, targets)
        
    check_gradients(func_logits, logits, d_logits)
