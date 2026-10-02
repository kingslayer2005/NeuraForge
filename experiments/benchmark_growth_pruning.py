"""
benchmark_growth_pruning.py — Benchmark network growth and pruning.

Trains a network on MNIST under different regimes to track accuracy vs parameter count.
Regimes:
1. fixed-small: A 3-layer MLP with 32 hidden units.
2. fixed-large: A 3-layer MLP with 128 hidden units.
3. grow-from-small: Start with 32 hidden units, train a bit, grow to 128, finish training.
4. prune-from-large (weight-norm): Start with 128, train, prune down to 32 using weight norm, finish training.
5. prune-from-large (Taylor): Start with 128, train, prune down to 32 using Taylor expansion, finish training.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings

import matplotlib

matplotlib.use('Agg')
import numpy as np
import pandas as pd

from experiments.datasets import load_mnist
from neuraforge.activations import ReLU
from neuraforge.data import DataLoader, Standardizer, train_val_test_split
from neuraforge.growth import widen_layer
from neuraforge.layers import Dense
from neuraforge.losses import SoftmaxCrossEntropy
from neuraforge.model import Sequential
from neuraforge.optimizers import MomentumSGD
from neuraforge.pruning import prune_layer
from neuraforge.seed import seed_everything
from neuraforge.train import evaluate, fit

warnings.filterwarnings("ignore")

def build_model(hidden_units: int) -> Sequential:
    return Sequential(
        Dense(784, hidden_units),
        ReLU(),
        Dense(hidden_units, hidden_units),
        ReLU(),
        Dense(hidden_units, 10)
    )

def count_parameters(model: Sequential) -> int:
    return sum(p.data.size for p in model.parameters())

def run_experiment(n_seeds: int = 5, epochs: int = 10) -> pd.DataFrame:
    print("--- Starting Growth and Pruning Benchmark on MNIST ---")
    
    X, y = load_mnist()
    Y = np.eye(10)[y]
    
    regimes = [
        "fixed-small", "fixed-large", "grow-from-small", 
        "prune-from-large (weight-norm)", "prune-from-large (Taylor)"
    ]
    
    results = []
    
    out_dir = Path(__file__).parent.parent / "results" / "growth_pruning"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for regime in regimes:
        print(f"\nEvaluating {regime}...")
        test_accs = []
        param_counts = []
        
        for seed in range(n_seeds):
            seed_everything(seed)
            splits = train_val_test_split(X, Y, val_frac=0.15, test_frac=0.15, seed=seed)
            
            scaler = Standardizer()
            splits['X_train'] = scaler.fit_transform(splits['X_train'])
            splits['X_val'] = scaler.transform(splits['X_val'])
            splits['X_test'] = scaler.transform(splits['X_test'])
            
            train_loader = DataLoader(splits['X_train'], splits['Y_train'], batch_size=64, shuffle=True)
            val_loader = DataLoader(splits['X_val'], splits['Y_val'], batch_size=64, shuffle=False)
            test_loader = DataLoader(splits['X_test'], splits['Y_test'], batch_size=64, shuffle=False)
            
            # Initial architecture
            initial_hidden = 128 if "large" in regime else 32
            model = build_model(initial_hidden)
            opt = MomentumSGD(model.parameters(), lr=0.01)
            loss_fn = SoftmaxCrossEntropy()
            
            # Train first half
            fit(model, opt, loss_fn, train_loader, val_loader, epochs=epochs//2, verbose=False)
            
            # Modify architecture if needed
            if regime == "grow-from-small":
                # Widen both hidden layers from 32 to 128
                widen_layer(model.layers[0], model.layers[2], new_neurons=128-32, method="random_noise")
                widen_layer(model.layers[2], model.layers[4], new_neurons=128-32, method="random_noise")
                # Need to recreate optimizer since parameters changed
                opt = MomentumSGD(model.parameters(), lr=0.01)
            elif regime == "prune-from-large (weight-norm)":
                # Prune both layers from 128 to 32
                prune_fraction = (128 - 32) / 128.0
                prune_layer(model.layers[0], model.layers[2], fraction=prune_fraction, method="magnitude")
                prune_layer(model.layers[2], model.layers[4], fraction=prune_fraction, method="magnitude")
                opt = MomentumSGD(model.parameters(), lr=0.01)
            elif regime == "prune-from-large (Taylor)":
                prune_fraction = (128 - 32) / 128.0
                prune_layer(model.layers[0], model.layers[2], fraction=prune_fraction, method="taylor")
                prune_layer(model.layers[2], model.layers[4], fraction=prune_fraction, method="taylor")
                opt = MomentumSGD(model.parameters(), lr=0.01)
                
            # Train second half
            fit(model, opt, loss_fn, train_loader, val_loader, epochs=epochs - epochs//2, verbose=False)
            
            # Evaluate
            _, test_acc = evaluate(model, loss_fn, test_loader)
            test_accs.append(test_acc)
            param_counts.append(count_parameters(model))
            
        results.append({
            "Regime": regime,
            "Parameters": np.mean(param_counts),
            "Test_Acc_Mean": np.mean(test_accs),
            "Test_Acc_Std": np.std(test_accs)
        })
        print(f"  Result: {np.mean(test_accs)*100:.2f}% ± {np.std(test_accs)*100:.2f}% (Params: {np.mean(param_counts):.0f})")
        
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "summary.csv", index=False)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    
    run_experiment(args.seeds, args.epochs)
