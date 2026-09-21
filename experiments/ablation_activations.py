"""
ablation_activations.py — Benchmark ForageAct against standard activations.

Trains a 3-layer MLP on MNIST and Fashion-MNIST with:
- ReLU, GELU, SiLU
- ForageAct (alpha=0.1, mode="fixed")
- ForageAct (alpha=0.1, mode="scalar")
- ForageAct (alpha=0.1, mode="per_neuron")

Learning rates are tuned on the validation set for each activation.
Runs 5 seeds and aggregates the results into CSV.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings

import matplotlib
matplotlib.use('Agg')  # Unattended runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiments.path_hack
from experiments.datasets import load_fashion_mnist, load_mnist
from neuraforge.activations import GELU, ReLU, SiLU, ForageAct
from neuraforge.data import DataLoader, Standardizer, train_val_test_split
from neuraforge.layers import Dense
from neuraforge.losses import SoftmaxCrossEntropy
from neuraforge.model import Sequential
from neuraforge.optimizers import NeuroGrad
from neuraforge.seed import seed_everything
from neuraforge.train import EarlyStopping, evaluate, fit

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def build_model(act_name: str, in_features: int = 784, hidden_features: int = 128, out_features: int = 10) -> Sequential:
    """Build a 3-layer MLP with the specified activation."""
    if act_name == "ReLU":
        act1 = ReLU()
        act2 = ReLU()
    elif act_name == "GELU":
        act1 = GELU()
        act2 = GELU()
    elif act_name == "SiLU":
        act1 = SiLU()
        act2 = SiLU()
    elif act_name == "ForageAct_Fixed":
        act1 = ForageAct(mode="fixed", init_alpha=0.1)
        act2 = ForageAct(mode="fixed", init_alpha=0.1)
    elif act_name == "ForageAct_Scalar":
        act1 = ForageAct(mode="scalar", init_alpha=0.1)
        act2 = ForageAct(mode="scalar", init_alpha=0.1)
    elif act_name == "ForageAct_PerNeuron":
        act1 = ForageAct(mode="per_neuron", init_alpha=0.1, n_neurons=hidden_features)
        act2 = ForageAct(mode="per_neuron", init_alpha=0.1, n_neurons=hidden_features)
    else:
        raise ValueError(f"Unknown activation: {act_name}")

    return Sequential(
        Dense(in_features, hidden_features),
        act1,
        Dense(hidden_features, hidden_features),
        act2,
        Dense(hidden_features, out_features)
    )


def run_experiment(dataset_name: str, n_seeds: int = 5, epochs: int = 50) -> pd.DataFrame:
    print(f"--- Starting Activation Ablation on {dataset_name} ---")
    
    if dataset_name == "MNIST":
        X, y = load_mnist()
    elif dataset_name == "Fashion-MNIST":
        X, y = load_fashion_mnist()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    # Standard 10-class one-hot
    Y = np.eye(10)[y]
    
    # We'll use a subset to keep the experiment runtime reasonable for 5 seeds * 6 acts * 3 LRs = 90 runs
    # 20k samples is enough to see learning differences without taking hours on CPU
    subset_size = 20000
    if X.shape[0] > subset_size:
        np.random.seed(42)
        idx = np.random.choice(X.shape[0], subset_size, replace=False)
        X, Y = X[idx], Y[idx]

    activations = ["ReLU", "GELU", "SiLU", "ForageAct_Fixed", "ForageAct_Scalar", "ForageAct_PerNeuron"]
    learning_rates = [0.05, 0.01, 0.005]  # Tune LR for each activation
    
    results = []
    
    # Base output dir
    out_dir = Path(__file__).parent.parent / "results" / "activation_ablation" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for act_name in activations:
        print(f"\nEvaluating {act_name}...")
        
        # 1. Tuning phase (Seed 42)
        print("  Tuning learning rate on validation set...")
        best_lr = learning_rates[0]
        best_val_loss = float('inf')
        
        for lr in learning_rates:
            seed_everything(42)
            splits = train_val_test_split(X, Y, val_frac=0.15, test_frac=0.15, seed=42)
            
            scaler = Standardizer()
            splits['X_train'] = scaler.fit_transform(splits['X_train'])
            splits['X_val'] = scaler.transform(splits['X_val'])
            
            train_loader = DataLoader(splits['X_train'], splits['Y_train'], batch_size=64, shuffle=True)
            val_loader = DataLoader(splits['X_val'], splits['Y_val'], batch_size=64, shuffle=False)
            
            model = build_model(act_name)
            opt = NeuroGrad(model.parameters(), lr=lr)
            loss_fn = SoftmaxCrossEntropy()
            
            # Short run for tuning
            history = fit(
                model, opt, loss_fn, train_loader, val_loader, 
                epochs=5, verbose=False
            )
            val_loss = history['val_loss'][-1]
            print(f"    LR {lr}: val_loss = {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_lr = lr
                
        print(f"  Best LR for {act_name}: {best_lr}")
        
        # 2. Evaluation phase (multiple seeds)
        test_accs = []
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
            
            model = build_model(act_name)
            opt = NeuroGrad(model.parameters(), lr=best_lr)
            loss_fn = SoftmaxCrossEntropy()
            es = EarlyStopping(patience=5, min_delta=0.001)
            
            # Alpha tracking for ForageAct
            track_alpha = "ForageAct" in act_name and "Fixed" not in act_name
            alpha_history = []
            
            start_time = time.time()
            for ep in range(epochs):
                if track_alpha:
                    # Layer 1 is the first activation
                    alpha_history.append(np.linalg.norm(model.layers[1].alpha))
                
                hist = fit(model, opt, loss_fn, train_loader, val_loader, epochs=1, verbose=False)
                val_loss = hist['val_loss'][0]
                
                if es(val_loss, model):
                    break
                    
            run_time = time.time() - start_time
            es.restore_best_weights(model)
            
            _, test_acc = evaluate(model, loss_fn, test_loader)
            test_accs.append(test_acc)
            
            # Plot alpha evolution for the first seed
            if track_alpha and seed == 0:
                plt.figure()
                plt.plot(alpha_history)
                plt.title(f"Alpha Norm Evolution ({act_name})")
                plt.xlabel("Epoch")
                plt.ylabel("L2 Norm of Alpha")
                plt.grid(True)
                plt.savefig(out_dir / f"{act_name}_alpha_evolution.png")
                plt.close()
                
        test_accs = np.array(test_accs)
        results.append({
            "Dataset": dataset_name,
            "Activation": act_name,
            "Best_LR": best_lr,
            "Test_Acc_Mean": np.mean(test_accs),
            "Test_Acc_Std": np.std(test_accs)
        })
        print(f"  Result: {np.mean(test_accs)*100:.2f}% ± {np.std(test_accs)*100:.2f}%")
        
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "summary.csv", index=False)
    
    # Generate a bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df["Activation"], df["Test_Acc_Mean"], yerr=df["Test_Acc_Std"], capsize=5)
    plt.title(f"Activation Performance on {dataset_name}")
    plt.ylabel("Test Accuracy")
    plt.xticks(rotation=45)
    plt.ylim(0.8, 1.0)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval - 0.05, f"{yval*100:.1f}%", ha='center', va='bottom', color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "performance_bar.png")
    plt.close()
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "Fashion-MNIST"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    
    run_experiment(args.dataset, args.seeds, args.epochs)
