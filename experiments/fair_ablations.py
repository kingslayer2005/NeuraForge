"""
fair_ablations.py — Phase 6 Fair Ablations

(1) ForageAct vs parameter-matched learnable baselines.
    Compares fixed activations (ReLU, GELU, SiLU) with learnable ones
    (PReLU, Swish, ForageAct) on an even footing.
    
(2) NeuroGrad vs decomposed variants.
    Separates the effects of EMA momentum vs gradient clipping.
"""

import sys
from pathlib import Path
import json
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from neuraforge.autograd import Tensor, softmax_cross_entropy, no_grad
from neuraforge.nn import Dense, Module
from neuraforge.layers import Parameter
from neuraforge.model import Sequential
from neuraforge.optimizers import SGD, MomentumSGD, Adam, NeuroGrad
from neuraforge.seed import seed_everything
from neuraforge.data import DataLoader, train_val_test_split
from experiments.datasets import load_mnist

def get_mnist_data():
    X, y = load_mnist()
    X = X / 255.0
    splits = train_val_test_split(X, y, val_frac=0.1, test_frac=0.1, seed=42)
    return splits["X_train"], splits["Y_train"], splits["X_test"], splits["Y_test"]


# ===================================================================
# CUSTOM ACTIVATIONS FOR ABLATION
# ===================================================================

from neuraforge.nn import ReLU, GELU, SiLU, ForageAct

class Mish(Module):
    def forward(self, x, **kwargs):
        self._x = x
        sp = np.log1p(np.exp(np.clip(x, -20, 20)))
        return x * np.tanh(sp)
    def backward(self, d_out):
        x = self._x
        sp = np.log1p(np.exp(np.clip(x, -20, 20)))
        t = np.tanh(sp)
        sig = 1.0 / (1.0 + np.exp(-x))
        grad = t + x * sig * (1.0 - t**2)
        return d_out * grad

class PReLU(Module):
    def __init__(self, num_parameters=1):
        super().__init__()
        self.alpha = Parameter("alpha", np.full(num_parameters, 0.25))
    def forward(self, x, **kwargs):
        self._x = x
        return np.maximum(x, 0) + self.alpha.data * np.minimum(x, 0)
    def backward(self, d_out):
        x = self._x
        self.alpha.grad = np.sum(d_out * np.minimum(x, 0), axis=0)
        grad = np.where(x > 0, 1.0, self.alpha.data)
        return d_out * grad
    def parameters(self): return [self.alpha]

class SwishLearned(Module):
    def __init__(self, num_parameters=1):
        super().__init__()
        self.beta = Parameter("beta", np.ones(num_parameters))
    def forward(self, x, **kwargs):
        self._x = x
        self._sig = 1.0 / (1.0 + np.exp(-self.beta.data * x))
        return x * self._sig
    def backward(self, d_out):
        x = self._x
        b = self.beta.data
        sig = self._sig
        
        # d(Swish)/dx = beta * Swish(x) + sig(x) * (1 - beta * Swish(x))
        # wait, Swish(x, b) = x * sig(b*x)
        # d/dx = sig(b*x) + x * b * sig(b*x) * (1 - sig(b*x))
        dx = sig + x * b * sig * (1.0 - sig)
        
        # d/dbeta = x^2 * sig * (1 - sig)
        db = x * x * sig * (1.0 - sig)
        self.beta.grad = np.sum(d_out * db, axis=0)
        
        return d_out * dx
    def parameters(self): return [self.beta]

# ===================================================================
# UTILS
# ===================================================================

def build_mlp(act_cls, width=128, depth=2):
    layers = []
    in_dim = 784
    for i in range(depth):
        layers.append(Dense(in_dim, width))
        if act_cls == PReLU or act_cls == SwishLearned:
            layers.append(act_cls(width))
        elif act_cls == ForageAct:
            layers.append(act_cls(mode="per_neuron", n_neurons=width))
        else:
            layers.append(act_cls())
        in_dim = width
    layers.append(Dense(in_dim, 10))
    return Sequential(*layers)

def evaluate(model, dl):
    correct = 0
    total = 0
    for xb, yb in dl:
        out = model.forward(xb, training=False)
        preds = np.argmax(out, axis=1)
        # Handle one-hot yb
        if yb.ndim == 2: yb = np.argmax(yb, axis=1)
        correct += np.sum(preds == yb)
        total += len(yb)
    return correct / total

def train_epoch(model, opt, dl):
    for xb, yb in dl:
        out = model.forward(xb, training=True)
        # Use autograd
        from neuraforge.autograd import Tensor, softmax_cross_entropy
        
        logits_t = Tensor(out, requires_grad=True)
        loss_t = softmax_cross_entropy(logits_t, yb)
        
        opt.zero_grad()
        loss_t.backward(np.ones_like(loss_t.data))
        model.backward(logits_t.grad)
        
        # apply global clipping if needed (for optimizers that don't do it)
        # For this ablation, we implement clipping separately if requested
        if hasattr(opt, "clip_grad_norm"):
            opt.clip_grad_norm()
            
        opt.step()

# ===================================================================
# OPTIMIZERS FOR ABLATION
# ===================================================================
# We extend optimizers to support manual clipping

def clip_grad_norm_(params, max_norm):
    total_norm = np.sqrt(sum(np.sum(p.grad**2) for p in params if p.grad is not None))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            if p.grad is not None:
                p.grad *= clip_coef

def get_optimizer(name, params, lr):
    if name == "SGD":
        return SGD(params, lr=lr)
    elif name == "SGD_momentum":
        return MomentumSGD(params, lr=lr, beta=0.9)
    elif name == "SGD_momentum_clip":
        opt = MomentumSGD(params, lr=lr, beta=0.9)
        opt.clip_grad_norm = lambda: clip_grad_norm_(params, 1.0)
        return opt
    elif name == "Adam":
        return Adam(params, lr=lr)
    elif name == "Adam_clip":
        opt = Adam(params, lr=lr)
        opt.clip_grad_norm = lambda: clip_grad_norm_(params, 1.0)
        return opt
    elif name == "NeuroGrad_noclip":
        return NeuroGrad(params, lr=lr, clip_mode="none")
    elif name == "NeuroGrad_full":
        return NeuroGrad(params, lr=lr, clip_mode="global_norm")
    else:
        raise ValueError(name)

# ===================================================================
# MAIN
# ===================================================================

def run_activation_ablation(out_dir):
    print("--- Running Activation Ablation ---")
    train_x, train_y, test_x, test_y = get_mnist_data()
    
    # We'll use one-hot for targets since autograd's softmax_cross_entropy expects one-hot
    # wait, load_mnist returns int labels for targets?
    # let's one-hot encode targets inside the DataLoader or here
    train_y_oh = np.zeros((train_y.size, 10))
    train_y_oh[np.arange(train_y.size), train_y] = 1.0
    
    test_y_oh = np.zeros((test_y.size, 10))
    test_y_oh[np.arange(test_y.size), test_y] = 1.0
    
    train_dl = DataLoader(train_x, train_y_oh, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_x, test_y_oh, batch_size=256, shuffle=False)

    
    activations = {
        "ReLU": ReLU, "GELU": GELU, "SiLU": SiLU, "Mish": Mish,
        "PReLU": PReLU, "SwishLearned": SwishLearned, "ForageAct": ForageAct
    }
    
    results = {}
    for name, act_cls in activations.items():
        print(f"Testing {name}...")
        accs = []
        for seed in range(3):
            seed_everything(seed)
            model = build_mlp(act_cls)
            params = model.parameters()
            n_params = sum(p.data.size for p in params)
            
            # Use Adam for all, tune lr implicitly (we just use 1e-3 here for simplicity, 
            # ideally we'd sweep but we have limited compute)
            opt = Adam(params, lr=1e-3)
            
            for epoch in range(3):
                train_epoch(model, opt, train_dl)
                
            acc = evaluate(model, test_dl)
            accs.append(acc)
            
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        results[name] = {"acc_mean": mean_acc, "acc_std": std_acc, "params": n_params}
        print(f"  {name}: {mean_acc*100:.2f}% ± {std_acc*100:.2f} (Params: {n_params})")
        
    with open(out_dir / "activation_ablation.json", "w") as f:
        json.dump(results, f, indent=2)

def run_optimizer_ablation(out_dir):
    print("--- Running Optimizer Ablation ---")
    train_x, train_y, test_x, test_y = get_mnist_data()
    
    train_y_oh = np.zeros((train_y.size, 10))
    train_y_oh[np.arange(train_y.size), train_y] = 1.0
    
    test_y_oh = np.zeros((test_y.size, 10))
    test_y_oh[np.arange(test_y.size), test_y] = 1.0
    
    train_dl = DataLoader(train_x, train_y_oh, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_x, test_y_oh, batch_size=256, shuffle=False)
    
    opts = [
        ("SGD", 0.1), ("SGD_momentum", 0.01), ("SGD_momentum_clip", 0.01),
        ("Adam", 1e-3), ("Adam_clip", 1e-3),
        ("NeuroGrad_noclip", 0.01), ("NeuroGrad_full", 0.01)
    ]
    
    results = {}
    for opt_name, lr in opts:
        print(f"Testing {opt_name} (lr={lr})...")
        accs = []
        for seed in range(3):
            seed_everything(seed)
            model = build_mlp(ReLU) # Fixed act for opt ablation
            params = model.parameters()
            opt = get_optimizer(opt_name, params, lr)
            
            for epoch in range(3):
                train_epoch(model, opt, train_dl)
                
            acc = evaluate(model, test_dl)
            accs.append(acc)
            
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        results[opt_name] = {"acc_mean": mean_acc, "acc_std": std_acc, "lr": lr}
        print(f"  {opt_name}: {mean_acc*100:.2f}% ± {std_acc*100:.2f}")
        
    with open(out_dir / "optimizer_ablation.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    out_dir = Path(__file__).parent.parent / "results" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    run_activation_ablation(out_dir)
    run_optimizer_ablation(out_dir)
