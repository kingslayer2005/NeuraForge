import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.datasets import load_fashion_mnist, load_mnist
from experiments.fair_ablations import (
    GELU,
    ForageAct,
    Mish,
    PReLU,
    ReLU,
    SiLU,
    SwishLearned,
    build_mlp,
    evaluate,
    get_optimizer,
    train_epoch,
)
from neuraforge.data import DataLoader, Standardizer, train_val_test_split
from neuraforge.seed import seed_everything


def prepare_data(dataset_name):
    if dataset_name == "MNIST":
        X, y = load_mnist()
    else:
        X, y = load_fashion_mnist()
        
    Y = np.zeros((y.size, 10))
    Y[np.arange(y.size), y] = 1.0
    
    splits = train_val_test_split(X, Y, val_frac=0.1, test_frac=0.1, seed=42)
    scaler = Standardizer()
    X_train = scaler.fit_transform(splits["X_train"])
    X_val = scaler.transform(splits["X_val"])
    X_test = scaler.transform(splits["X_test"])
    
    return {
        "train_dl": DataLoader(X_train, splits["Y_train"], batch_size=128, shuffle=True),
        "val_dl": DataLoader(X_val, splits["Y_val"], batch_size=256, shuffle=False),
        "test_dl": DataLoader(X_test, splits["Y_test"], batch_size=256, shuffle=False)
    }

def run_activation_ablation(out_dir):
    print("--- Running Activation Ablation V2 ---")
    
    activations = {
        "ReLU": ReLU, "GELU": GELU, "SiLU": SiLU, "Mish": Mish,
        "PReLU": PReLU, "SwishLearned": SwishLearned, "ForageAct": ForageAct
    }
    
    lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    
    out_file = out_dir / "activation_ablation_v2.json"
    results = {}
    if out_file.exists():
        try:
            with open(out_file, "r") as f:
                results = json.load(f)
        except json.JSONDecodeError:
            pass

    for ds_name in ["MNIST", "Fashion-MNIST"]:
        print(f"\nDataset: {ds_name}")
        data = prepare_data(ds_name)
        if ds_name not in results:
            results[ds_name] = {}
        
        for act_name, act_cls in activations.items():
            if act_name in results[ds_name]:
                print(f"  Skipping {act_name} (already computed)")
                continue
            
            print(f"  Testing {act_name}...")
            
            # Tune LR on validation set (Seed 0)
            best_val_acc = -1
            best_lr = lrs[0]
            
            for lr in lrs:
                seed_everything(0)
                model = build_mlp(act_cls)
                opt = get_optimizer("Adam", model.parameters(), lr)
                for _ in range(2): # Quick 2 epochs for tuning
                    train_epoch(model, opt, data["train_dl"])
                val_acc = evaluate(model, data["val_dl"])
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_lr = lr
                    
            print(f"    Selected LR: {best_lr} (Val Acc: {best_val_acc*100:.2f}%)")
            
            # Run 5 seeds with best LR
            test_accs = []
            for seed in range(5):
                seed_everything(seed)
                model = build_mlp(act_cls)
                opt = get_optimizer("Adam", model.parameters(), best_lr)
                for _ in range(3): # 3 epochs for full run
                    train_epoch(model, opt, data["train_dl"])
                test_acc = evaluate(model, data["test_dl"])
                test_accs.append(test_acc)
            
            mean_acc = np.mean(test_accs)
            std_acc = np.std(test_accs)
            ci_95 = 1.96 * std_acc / np.sqrt(5)
            
            params = sum(p.data.size for p in model.parameters())
            
            res = {
                "best_lr": best_lr,
                "acc_mean": mean_acc,
                "acc_ci95": ci_95,
                "params": params
            }
            results[ds_name][act_name] = res
            print(f"    Final Test Acc: {mean_acc*100:.2f}% ± {ci_95*100:.2f}% (Params: {params})")
            
            with open(out_dir / "activation_ablation_v2.json", "w") as f:
                json.dump(results, f, indent=2)


def run_optimizer_ablation(out_dir):
    print("--- Running Optimizer Ablation V2 ---")
    
    opt_variants = [
        "SGD", "SGD_momentum", "SGD_momentum_clip",
        "Adam", "Adam_clip",
        "NeuroGrad_noclip", "NeuroGrad_full"
    ]
    
    lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    
    out_file = out_dir / "optimizer_ablation_v2.json"
    results = {}
    if out_file.exists():
        try:
            with open(out_file, "r") as f:
                results = json.load(f)
        except json.JSONDecodeError:
            pass

    for ds_name in ["MNIST", "Fashion-MNIST"]:
        print(f"\nDataset: {ds_name}")
        data = prepare_data(ds_name)
        if ds_name not in results:
            results[ds_name] = {}
        
        for opt_name in opt_variants:
            if opt_name in results[ds_name]:
                print(f"  Skipping {opt_name} (already computed)")
                continue
            
            print(f"  Testing {opt_name}...")
            
            best_val_acc = -1
            best_lr = lrs[0]
            
            for lr in lrs:
                seed_everything(0)
                model = build_mlp(ReLU) # Standard ReLU network
                opt = get_optimizer(opt_name, model.parameters(), lr)
                for _ in range(2): 
                    train_epoch(model, opt, data["train_dl"])
                val_acc = evaluate(model, data["val_dl"])
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_lr = lr
                    
            print(f"    Selected LR: {best_lr} (Val Acc: {best_val_acc*100:.2f}%)")
            
            test_accs = []
            for seed in range(5):
                seed_everything(seed)
                model = build_mlp(ReLU)
                opt = get_optimizer(opt_name, model.parameters(), best_lr)
                for _ in range(3): 
                    train_epoch(model, opt, data["train_dl"])
                test_acc = evaluate(model, data["test_dl"])
                test_accs.append(test_acc)
            
            mean_acc = np.mean(test_accs)
            std_acc = np.std(test_accs)
            ci_95 = 1.96 * std_acc / np.sqrt(5)
            
            res = {
                "best_lr": best_lr,
                "acc_mean": mean_acc,
                "acc_ci95": ci_95
            }
            results[ds_name][opt_name] = res
            print(f"    Final Test Acc: {mean_acc*100:.2f}% ± {ci_95*100:.2f}%")
            
            with open(out_dir / "optimizer_ablation_v2.json", "w") as f:
                json.dump(results, f, indent=2)

def main():
    root = Path(__file__).parent.parent
    out_dir = root / "results" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    run_activation_ablation(out_dir)
    run_optimizer_ablation(out_dir)

if __name__ == "__main__":
    main()
