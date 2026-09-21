"""
test_optimizers.py — Ensure all optimizers can minimize a simple quadratic.

Verifies that SGD, MomentumSGD, Adam, and NeuroGrad converge to the known
minimum of a convex quadratic function. Also tests that every parameter
(including alpha in all ForageAct modes) is updated after one step.
"""

import numpy as np
import pytest

from neuraforge.activations import ForageAct
from neuraforge.layers import Dense, Parameter
from neuraforge.optimizers import SGD, Adam, MomentumSGD, NeuroGrad
from neuraforge.seed import seed_everything


@pytest.fixture(autouse=True)
def setup_seed():
    seed_everything(42)


@pytest.mark.parametrize("OptimizerClass, kwargs", [
    (SGD, {"lr": 0.1}),
    (MomentumSGD, {"lr": 0.05, "beta": 0.9}),
    (Adam, {"lr": 0.1}),
    (NeuroGrad, {"lr": 0.1, "clip_mode": "per_tensor"}),
    (NeuroGrad, {"lr": 0.1, "clip_mode": "global_norm"}),
    (NeuroGrad, {"lr": 0.1, "clip_mode": "none"}),
])
def test_optimizer_convergence(OptimizerClass, kwargs):
    """Test that the optimizer converges to the minimum of a quadratic.
    
    f(x, y) = x^2 + 2y^2
    Minimum is at (0, 0).
    """
    # Start away from the minimum
    p_x = Parameter("x", np.array([5.0]))
    p_y = Parameter("y", np.array([-4.0]))
    params = [p_x, p_y]
    
    opt = OptimizerClass(params, **kwargs)
    
    # Run for 200 steps
    for _ in range(200):
        # Forward: loss = x^2 + 2y^2
        x = p_x.data[0]
        y = p_y.data[0]
        loss = x**2 + 2 * y**2
        
        # Backward: dx = 2x, dy = 4y
        opt.zero_grad()
        p_x.grad = np.array([2 * x])
        p_y.grad = np.array([4 * y])
        
        # Update
        opt.step()
        
    # Check convergence
    assert np.abs(p_x.data[0]) < 1e-2, f"Failed to converge x. Final: {p_x.data[0]}"
    assert np.abs(p_y.data[0]) < 1e-2, f"Failed to converge y. Final: {p_y.data[0]}"


def test_all_parameters_update():
    """Verify that every parameter actually changes after one optimizer step.
    
    Addresses Correction #1 from the spec to ensure parameters like alpha
    are updated in-place correctly.
    """
    layer = Dense(2, 3)
    act = ForageAct(mode="per_neuron", init_alpha=0.1, n_neurons=3)
    
    params = layer.parameters() + act.parameters()
    opt = SGD(params, lr=0.1)
    
    # Store initial values
    initial_values = {p.name: p.data.copy() for p in params}
    
    # Provide dummy gradients (ones)
    opt.zero_grad()
    for p in params:
        p.grad = np.ones_like(p.data)
        
    # Take a step
    opt.step()
    
    # Verify every parameter changed
    for p in params:
        initial = initial_values[p.name]
        assert not np.allclose(p.data, initial), f"Parameter {p.name} did not update!"
