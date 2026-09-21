"""
benchmark_performance.py — Benchmark NeuraForge against PyTorch CPU.

Tests the float32 and float64 paths for an identical MLP architecture
on both NeuraForge and PyTorch. Honest measurement of forward/backward time.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from neuraforge.model import Sequential
from neuraforge.layers import Dense
from neuraforge.activations import ReLU
from neuraforge.losses import SoftmaxCrossEntropy
from neuraforge.seed import seed_everything


class TorchMLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_features, out_features)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x


def benchmark_frameworks():
    batch_size = 256
    in_dim = 784
    hidden_dim = 256
    out_dim = 10
    n_iters = 100
    
    results = []
    out_dir = Path(__file__).parent.parent / "results" / "performance"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for dtype_name, np_dtype, pt_dtype in [("float32", np.float32, torch.float32), ("float64", np.float64, torch.float64)]:
        print(f"\nBenchmarking {dtype_name}...")
        
        # 1. Setup PyTorch
        seed_everything(42)
        X_pt = torch.randn(batch_size, in_dim, dtype=pt_dtype, requires_grad=True)
        Y_pt = torch.randint(0, out_dim, (batch_size,))
        pt_model = TorchMLP(in_dim, hidden_dim, out_dim).to(pt_dtype)
        pt_loss_fn = nn.CrossEntropyLoss()
        
        # Warmup
        for _ in range(10):
            pt_loss_fn(pt_model(X_pt), Y_pt).backward()
            
        pt_model.zero_grad()
        X_pt.grad = None
        
        start = time.perf_counter()
        for _ in range(n_iters):
            loss = pt_loss_fn(pt_model(X_pt), Y_pt)
            loss.backward()
            pt_model.zero_grad()
            X_pt.grad = None
        pt_time = time.perf_counter() - start
        pt_ms_per_iter = (pt_time / n_iters) * 1000
        
        print(f"  PyTorch ({dtype_name}): {pt_ms_per_iter:.2f} ms / step")
        
        # 2. Setup NeuraForge
        seed_everything(42)
        X_np = np.random.randn(batch_size, in_dim).astype(np_dtype)
        Y_np = np.eye(out_dim)[np.random.randint(0, out_dim, batch_size)].astype(np_dtype)
        
        nf_model = Sequential(
            Dense(in_dim, hidden_dim),
            ReLU(),
            Dense(hidden_dim, hidden_dim),
            ReLU(),
            Dense(hidden_dim, out_dim)
        )
        
        # Cast parameters to correct dtype
        for p in nf_model.parameters():
            p.data = p.data.astype(np_dtype)
            
        nf_loss_fn = SoftmaxCrossEntropy()
        
        # Warmup
        for _ in range(10):
            out = nf_model.forward(X_np)
            loss = nf_loss_fn.forward(out, Y_np)
            nf_model.backward(nf_loss_fn.backward())
            
        start = time.perf_counter()
        for _ in range(n_iters):
            out = nf_model.forward(X_np)
            loss = nf_loss_fn.forward(out, Y_np)
            nf_model.backward(nf_loss_fn.backward())
        nf_time = time.perf_counter() - start
        nf_ms_per_iter = (nf_time / n_iters) * 1000
        
        print(f"  NeuraForge ({dtype_name}): {nf_ms_per_iter:.2f} ms / step")
        
        results.append({
            "Precision": dtype_name,
            "Framework": "PyTorch (CPU)",
            "Time (ms/iter)": pt_ms_per_iter
        })
        results.append({
            "Precision": dtype_name,
            "Framework": "NeuraForge",
            "Time (ms/iter)": nf_ms_per_iter
        })
        
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "benchmark.csv", index=False)
    
    # Plot results
    plt.figure(figsize=(8, 6))
    
    f32 = df[df["Precision"] == "float32"]
    f64 = df[df["Precision"] == "float64"]
    
    x = np.arange(2)
    width = 0.35
    
    plt.bar(x - width/2, f32["Time (ms/iter)"], width, label="float32")
    plt.bar(x + width/2, f64["Time (ms/iter)"], width, label="float64")
    
    plt.ylabel("Time (ms per forward+backward step)")
    plt.title("Performance Benchmark: NeuraForge vs PyTorch CPU")
    plt.xticks(x, ["PyTorch (CPU)", "NeuraForge"])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(out_dir / "performance.png")
    plt.close()

if __name__ == "__main__":
    benchmark_frameworks()
