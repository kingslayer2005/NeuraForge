"""
compare_optimizers.py — Benchmark standard optimizers vs NeuroGrad.

Trains a 3-layer MLP on MNIST and Fashion-MNIST with:
- SGD
- MomentumSGD
- Adam
- NeuroGrad (clip_mode="per_tensor")
- NeuroGrad (clip_mode="global_norm")
- NeuroGrad (clip_mode="none")

Learning rates are tuned on the validation set for each optimizer.
Runs 5 seeds and aggregates the results into CSV.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.datasets import load_fashion_mnist, load_mnist
from neuraforge.activations import ReLU
from neuraforge.data import DataLoader, Standardizer, train_val_test_split
from neuraforge.layers import Dense
from neuraforge.losses import SoftmaxCrossEntropy
from neuraforge.model import Sequential
from neuraforge.optimizers import SGD, Adam, MomentumSGD, NeuroGrad
from neuraforge.seed import seed_everything
from neuraforge.train import EarlyStopping, evaluate, fit

warnings.filterwarnings("ignore")


def build_model(in_features: int = 784, hidden_features: int = 128, out_features: int = 10) -> Sequential:
    return Sequential(
        Dense(in_features, hidden_features),
        ReLU(),
        Dense(hidden_features, hidden_features),
        ReLU(),
        Dense(hidden_features, out_features)
    )


def get_optimizer(opt_name: str, params: list, lr: float):
    if opt_name == "SGD":
        return SGD(params, lr=lr)
    elif opt_name == "Momentum":
        return MomentumSGD(params, lr=lr, beta=0.9)
    elif opt_name == "Adam":
        return Adam(params, lr=lr)
    elif opt_name == "NeuroGrad_PerTensor":
        return NeuroGrad(params, lr=lr, beta=0.9, clip_value=1.0, clip_mode="per_tensor")
    elif opt_name == "NeuroGrad_Global":
        return NeuroGrad(params, lr=lr, beta=0.9, clip_value=5.0, clip_mode="global_norm")
    elif opt_name == "NeuroGrad_NoClip":
        return NeuroGrad(params, lr=lr, beta=0.9, clip_mode="none")
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def run_experiment(dataset_name: str, n_seeds: int = 5, epochs: int = 50) -> pd.DataFrame:
    print(f"--- Starting Optimizer Comparison on {dataset_name} ---")
    
    if dataset_name == "MNIST":
        X, y = load_mnist()
    elif dataset_name == "Fashion-MNIST":
        X, y = load_fashion_mnist()
        
    Y = np.eye(10)[y]
    
    # Use full dataset
    # subset_size = 20000
    # if X.shape[0] > subset_size:
    #     np.random.seed(42)
    #     idx = np.random.choice(X.shape[0], subset_size, replace=False)
    #     X, Y = X[idx], Y[idx]

    optimizers = [
        "SGD", "Momentum", "Adam", 
        "NeuroGrad_PerTensor", "NeuroGrad_Global", "NeuroGrad_NoClip"
    ]
    
    # Different optimizers need vastly different LR ranges
    lr_grid = {
        "SGD": [0.1, 0.05, 0.01],
        "Momentum": [0.05, 0.01, 0.005],
        "Adam": [0.005, 0.001, 0.0005],
        "NeuroGrad_PerTensor": [0.1, 0.05, 0.01],
        "NeuroGrad_Global": [0.1, 0.05, 0.01],
        "NeuroGrad_NoClip": [0.1, 0.05, 0.01],
    }
    
    results = []
    
    out_dir = Path(__file__).parent.parent / "results" / "optimizer_comparison" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # We will plot the loss curve of the best seed for each optimizer
    best_loss_curves = {}
    
    for opt_name in optimizers:
        print(f"\nEvaluating {opt_name}...")
        
        # 1. Tuning phase
        best_lr = lr_grid[opt_name][0]
        best_val_loss = float('inf')
        
        for lr in lr_grid[opt_name]:
            seed_everything(42)
            splits = train_val_test_split(X, Y, val_frac=0.15, test_frac=0.15, seed=42)
            scaler = Standardizer()
            splits['X_train'] = scaler.fit_transform(splits['X_train'])
            splits['X_val'] = scaler.transform(splits['X_val'])
            
            train_loader = DataLoader(splits['X_train'], splits['Y_train'], batch_size=64, shuffle=True)
            val_loader = DataLoader(splits['X_val'], splits['Y_val'], batch_size=64, shuffle=False)
            
            model = build_model()
            opt = get_optimizer(opt_name, model.parameters(), lr)
            loss_fn = SoftmaxCrossEntropy()
            
            history = fit(model, opt, loss_fn, train_loader, val_loader, epochs=3, verbose=False)
            val_loss = history['val_loss'][-1]
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_lr = lr
                
        print(f"  Best LR for {opt_name}: {best_lr}")
        
        # 2. Evaluation phase
        test_accs = []
        best_run_val_loss = float('inf')
        
        for seed in range(n_seeds):
            print(f"  Running seed {seed+1}/{n_seeds}...")
            seed_everything(seed)
            splits = train_val_test_split(X, Y, val_frac=0.15, test_frac=0.15, seed=seed)
            
            scaler = Standardizer()
            splits['X_train'] = scaler.fit_transform(splits['X_train'])
            splits['X_val'] = scaler.transform(splits['X_val'])
            splits['X_test'] = scaler.transform(splits['X_test'])
            
            train_loader = DataLoader(splits['X_train'], splits['Y_train'], batch_size=64, shuffle=True)
            val_loader = DataLoader(splits['X_val'], splits['Y_val'], batch_size=64, shuffle=False)
            test_loader = DataLoader(splits['X_test'], splits['Y_test'], batch_size=64, shuffle=False)
            
            model = build_model()
            opt = get_optimizer(opt_name, model.parameters(), best_lr)
            loss_fn = SoftmaxCrossEntropy()
            es = EarlyStopping(patience=5, min_delta=0.001)
            
            hist = fit(model, opt, loss_fn, train_loader, val_loader, epochs=epochs, early_stopping=es, verbose=False)
            
            _, test_acc = evaluate(model, loss_fn, test_loader)
            test_accs.append(test_acc)
            
            # Save the training curve for the best run
            final_val = min(hist['val_loss'])
            if final_val < best_run_val_loss:
                best_run_val_loss = final_val
                best_loss_curves[opt_name] = hist['val_loss']
                
        test_accs = np.array(test_accs)
        results.append({
            "Dataset": dataset_name,
            "Optimizer": opt_name,
            "Best_LR": best_lr,
            "Test_Acc_Mean": np.mean(test_accs),
            "Test_Acc_Std": np.std(test_accs)
        })
        print(f"  Result: {np.mean(test_accs)*100:.2f}% ± {np.std(test_accs)*100:.2f}%")
        
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "summary.csv", index=False)
    
    # Generate convergence plot comparing optimizers
    plt.figure(figsize=(10, 6))
    for opt_name, curve in best_loss_curves.items():
        plt.plot(curve, label=opt_name, linewidth=2)
    plt.title(f"Optimizer Convergence on {dataset_name} (Validation Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / "convergence_comparison.png")
    plt.close()
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "Fashion-MNIST"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    
    run_experiment(args.dataset, args.seeds, args.epochs)
