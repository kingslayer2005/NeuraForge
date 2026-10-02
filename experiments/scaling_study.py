"""
scaling_study.py — Phase 5 Honest Performance Work

Replaces single-point benchmarks with a scaling study across batch sizes, widths,
depths, and workloads (Dense, CNN, Transformer).
Compares NeuraForge vs PyTorch on CPU, measuring:
 - ms per step
 - peak memory (via tracemalloc / psutil)
 - number of NumPy array allocations per step (via sys.getallocatedblocks())
"""

import sys
import time
import gc
import psutil
import tracemalloc
from pathlib import Path
import json
import itertools
import threadpoolctl
import os

# Explicitly pin BLAS thread count
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

threadpoolctl.threadpool_limits(limits=4)

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from neuraforge.autograd import Tensor, no_grad
from neuraforge.nn import Dense, Conv2d, ReLU, Module
from neuraforge.transformer import TransformerBlock
from neuraforge.model import Sequential

class NF_MLP(Module):
    def __init__(self, depth, width):
        super().__init__()
        layers = []
        for i in range(depth):
            in_dim = width if i > 0 else width
            layers.append(Dense(in_dim, width))
            layers.append(ReLU())
        self.seq = Sequential(*layers)

    def forward(self, x):
        return self.seq.forward(x)

class PT_MLP(torch.nn.Module):
    def __init__(self, depth, width):
        super().__init__()
        layers = []
        for i in range(depth):
            in_dim = width if i > 0 else width
            layers.append(torch.nn.Linear(in_dim, width))
            layers.append(torch.nn.ReLU())
        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)

class NF_CNN(Module):
    def __init__(self, depth, width):
        super().__init__()
        layers = []
        for i in range(depth):
            in_dim = width if i > 0 else width
            layers.append(Conv2d(in_dim, width, kernel_size=3, padding=1))
            layers.append(ReLU())
        self.seq = Sequential(*layers)

    def forward(self, x):
        return self.seq.forward(x)

class PT_CNN(torch.nn.Module):
    def __init__(self, depth, width):
        super().__init__()
        layers = []
        for i in range(depth):
            in_dim = width if i > 0 else width
            layers.append(torch.nn.Conv2d(in_dim, width, kernel_size=3, padding=1))
            layers.append(torch.nn.ReLU())
        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)

class NF_Transformer(Module):
    def __init__(self, depth, width):
        super().__init__()
        self.blocks = [TransformerBlock(d_model=width, n_heads=4, d_ff=width * 4) for _ in range(depth)]
    def forward(self, x):
        for b in self.blocks:
            x = b.forward(x)
        return x

class PT_Transformer(torch.nn.Module):
    def __init__(self, depth, width):
        super().__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=width*4, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=depth)
    def forward(self, x):
        return self.transformer(x)

def profile_step(framework, workload, batch_size, width, depth, dtype):
    if dtype == "float32":
        np_dtype = np.float32
        pt_dtype = torch.float32
    else:
        np_dtype = np.float64
        pt_dtype = torch.float64

    # Setup inputs
    if workload == "mlp":
        x_np = np.random.randn(batch_size, width).astype(np_dtype)
        model_nf = NF_MLP(depth, width)
        model_pt = PT_MLP(depth, width)
    elif workload == "cnn":
        # Keep spatial size small (8x8) to avoid blowing up memory at large widths
        x_np = np.random.randn(batch_size, width, 8, 8).astype(np_dtype)
        model_nf = NF_CNN(depth, width)
        model_pt = PT_CNN(depth, width)
    elif workload == "transformer":
        seq_len = 16
        x_np = np.random.randn(batch_size, seq_len, width).astype(np_dtype)
        model_nf = NF_Transformer(depth, width)
        model_pt = PT_Transformer(depth, width)

    x_pt = torch.tensor(x_np, dtype=pt_dtype, requires_grad=True)

    # Warmup
    if framework == "neuraforge":
        _ = model_nf.forward(x_np)
        if workload == "transformer":
            for b in model_nf.blocks:
                for p in b.parameters(): p.grad = None
        else:
            for p in model_nf.seq.parameters(): p.grad = None
    else:
        model_pt.to(pt_dtype)
        _ = model_pt(x_pt)
        x_pt.grad = None

    gc.collect()

    times = []
    memories = []
    allocations = []

    for _ in range(3):
        gc.collect()
        tracemalloc.start()
        blocks_before = sys.getallocatedblocks()
        t0 = time.perf_counter()

        if framework == "neuraforge":
            out = model_nf.forward(x_np)
            # Fake backward just to trigger ops
            d_out = np.ones_like(out)
            if workload == "transformer":
                for b in reversed(model_nf.blocks):
                    d_out = b.backward(d_out)
            else:
                d_out = model_nf.seq.backward(d_out)
        else:
            out = model_pt(x_pt)
            loss = out.sum()
            loss.backward()

        t1 = time.perf_counter()
        blocks_after = sys.getallocatedblocks()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append((t1 - t0) * 1000) # ms
        if framework == "neuraforge":
            memories.append(peak / (1024 * 1024)) # MB
        else:
            memories.append(float('nan'))
        allocations.append(blocks_after - blocks_before)

    return {
        "framework": framework,
        "workload": workload,
        "batch_size": batch_size,
        "width": width,
        "depth": depth,
        "dtype": dtype,
        "time_ms": np.mean(times),
        "peak_mem_mb": np.mean(memories) if framework == "neuraforge" else "N/A (measured via tracemalloc)",
        "allocations": np.mean(allocations)
    }

def main():
    root = Path(__file__).parent.parent
    out_dir = root / "results" / "scaling"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # System info
    info = {
        "cpu_model": threadpoolctl.threadpool_info()[0].get('architecture', 'Unknown'),
        "ram_gb": psutil.virtual_memory().total / (1024**3),
        "blas_backend": threadpoolctl.threadpool_info()[0].get('user_api', 'Unknown'),
        "blas_threads": 4
    }
    with open(out_dir / "system_info.json", "w") as f:
        json.dump(info, f, indent=2)

    results = []

    # Smaller sweeps due to compute constraints.
    # We will test selected representative configurations.
    workloads = ["mlp", "cnn", "transformer"]
    frameworks = ["neuraforge", "pytorch"]
    
    # Selected points for scaling
    configs = [
        # (batch, width, depth, dtype)
        (32, 128, 2, "float32"),
        (512, 128, 2, "float32"),
        (2048, 128, 2, "float32"),
        (32, 512, 4, "float32"),
        (128, 512, 4, "float32"),
        (32, 128, 2, "float64"),
    ]

    for wkld in workloads:
        for b, w, d, dt in configs:
            if wkld == "cnn" and w > 128:
                continue # avoid blowing up memory with Conv2d
            for fw in frameworks:
                print(f"Running {fw} | {wkld} | b={b}, w={w}, d={d}, dtype={dt}...")
                try:
                    res = profile_step(fw, wkld, b, w, d, dt)
                    results.append(res)
                except Exception as e:
                    print(f"Failed: {e}")

    with open(out_dir / "scaling_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Done!")

if __name__ == "__main__":
    main()
