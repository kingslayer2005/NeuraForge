"""
test_torch_parity.py — Compare NeuraForge exactly against PyTorch.

Verifies that NeuraForge produces the exact same forward outputs and
backward gradients as an equivalent PyTorch model for a 3-layer MLP.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F
from torch import nn

from neuraforge.activations import ForageAct
from neuraforge.layers import Dense
from neuraforge.losses import SoftmaxCrossEntropy
from neuraforge.model import Sequential
from neuraforge.seed import seed_everything


class TorchForageAct(nn.Module):
    """PyTorch implementation of ForageAct for testing."""
    def __init__(self, mode="scalar", init_alpha=0.1, n_neurons=None):
        super().__init__()
        self.mode = mode
        if mode == "fixed":
            self.alpha = init_alpha
        elif mode == "scalar":
            self.alpha = nn.Parameter(torch.tensor([init_alpha], dtype=torch.float64))
        elif mode == "per_neuron":
            self.alpha = nn.Parameter(torch.full((n_neurons,), init_alpha, dtype=torch.float64))
            
    def forward(self, z):
        return z * torch.sigmoid(z) + self.alpha * torch.tanh(z)


class TorchMLP(nn.Module):
    """Equivalent PyTorch MLP."""
    def __init__(self, in_features, hidden_features, out_features, act_mode="scalar"):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = TorchForageAct(mode=act_mode, init_alpha=0.1, n_neurons=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


def copy_weights_to_torch(nf_model, torch_model):
    """Copy weights from NeuraForge model to Torch model."""
    with torch.no_grad():
        # PyTorch stores weights as (out_features, in_features)
        torch_model.fc1.weight.copy_(torch.from_numpy(nf_model.layers[0].W.data.T))
        # PyTorch bias shape is (out_features,)
        torch_model.fc1.bias.copy_(torch.from_numpy(nf_model.layers[0].b.data.squeeze()))
        
        torch_model.fc2.weight.copy_(torch.from_numpy(nf_model.layers[2].W.data.T))
        torch_model.fc2.bias.copy_(torch.from_numpy(nf_model.layers[2].b.data.squeeze()))
        
        if nf_model.layers[1].mode != "fixed":
            torch_model.act1.alpha.copy_(torch.from_numpy(nf_model.layers[1].alpha))


@pytest.fixture(autouse=True)
def setup_seed():
    seed_everything(42)
    torch.manual_seed(42)


@pytest.mark.parametrize("act_mode", ["scalar", "per_neuron"])
def test_torch_parity_mlp(act_mode):
    batch_size, in_dim, hidden_dim, out_dim = 4, 10, 32, 3
    
    # 1. Generate random data
    X = np.random.randn(batch_size, in_dim).astype(np.float64)
    y_idx = np.random.randint(0, out_dim, size=(batch_size,))
    Y = np.eye(out_dim)[y_idx].astype(np.float64)
    
    X_torch = torch.tensor(X, requires_grad=True)
    y_torch = torch.tensor(y_idx, dtype=torch.long)
    
    # 2. Build NeuraForge model (float64 precision)
    nf_model = Sequential(
        Dense(in_dim, hidden_dim),
        ForageAct(mode=act_mode, init_alpha=0.1, n_neurons=hidden_dim),
        Dense(hidden_dim, out_dim)
    )
    # Ensure float64
    for p in nf_model.parameters():
        p.data = p.data.astype(np.float64)
        
    nf_loss_fn = SoftmaxCrossEntropy()
    
    # 3. Build PyTorch model
    torch_model = TorchMLP(in_dim, hidden_dim, out_dim, act_mode=act_mode).double()
    
    # 4. Sync weights
    copy_weights_to_torch(nf_model, torch_model)
    
    # 5. Forward Pass
    nf_logits = nf_model.forward(X)
    nf_loss = nf_loss_fn.forward(nf_logits, Y)
    
    torch_logits = torch_model(X_torch)
    # PyTorch CrossEntropyLoss takes raw logits and class indices
    torch_loss = F.cross_entropy(torch_logits, y_torch)
    
    np.testing.assert_allclose(
        nf_logits, torch_logits.detach().numpy(),
        rtol=1e-6, atol=1e-8, err_msg="Logits mismatch"
    )
    np.testing.assert_allclose(
        nf_loss, torch_loss.item(),
        rtol=1e-6, atol=1e-8, err_msg="Loss mismatch"
    )
    
    # 6. Backward Pass
    nf_d_out = nf_loss_fn.backward()
    nf_d_in = nf_model.backward(nf_d_out)
    
    torch_loss.backward()
    
    # 7. Check Gradients
    
    # Input gradient
    np.testing.assert_allclose(
        nf_d_in, X_torch.grad.numpy(),
        rtol=1e-6, atol=1e-8, err_msg="Input gradient mismatch"
    )
    
    # fc1 gradients
    np.testing.assert_allclose(
        nf_model.layers[0].W.grad, torch_model.fc1.weight.grad.numpy().T,
        rtol=1e-6, atol=1e-8, err_msg="fc1 W gradient mismatch"
    )
    np.testing.assert_allclose(
        nf_model.layers[0].b.grad.squeeze(), torch_model.fc1.bias.grad.numpy(),
        rtol=1e-6, atol=1e-8, err_msg="fc1 b gradient mismatch"
    )
    
    # act1 alpha gradient
    np.testing.assert_allclose(
        nf_model.layers[1].parameters()[0].grad, torch_model.act1.alpha.grad.numpy(),
        rtol=1e-6, atol=1e-8, err_msg="act1 alpha gradient mismatch"
    )
    
    # fc2 gradients
    np.testing.assert_allclose(
        nf_model.layers[2].W.grad, torch_model.fc2.weight.grad.numpy().T,
        rtol=1e-6, atol=1e-8, err_msg="fc2 W gradient mismatch"
    )
    np.testing.assert_allclose(
        nf_model.layers[2].b.grad.squeeze(), torch_model.fc2.bias.grad.numpy(),
        rtol=1e-6, atol=1e-8, err_msg="fc2 b gradient mismatch"
    )
