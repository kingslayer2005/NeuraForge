import json
from pathlib import Path

# Data from log:
#  Testing ReLU...
#    Selected LR: 0.001 (Val Acc: 96.20%)
#    Final Test Acc: 96.70%  0.09% (Params: 118282)
# ...

data = {
  "MNIST": {
    "ReLU": {
      "best_lr": 0.001,
      "acc_mean": 0.9670,
      "acc_ci95": 0.0009,
      "params": 118282
    },
    "GELU": {
      "best_lr": 0.001,
      "acc_mean": 0.9688,
      "acc_ci95": 0.0005,
      "params": 118282
    },
    "SiLU": {
      "best_lr": 0.001,
      "acc_mean": 0.9695,
      "acc_ci95": 0.0013,
      "params": 118282
    },
    "Mish": {
      "best_lr": 0.001,
      "acc_mean": 0.9691,
      "acc_ci95": 0.0010,
      "params": 118282
    },
    "PReLU": {
      "best_lr": 0.003,
      "acc_mean": 0.9661,
      "acc_ci95": 0.0010,
      "params": 118538
    },
    "SwishLearned": {
      "best_lr": 0.001,
      "acc_mean": 0.9697,
      "acc_ci95": 0.0008,
      "params": 118538
    },
    "ForageAct": {
      "best_lr": 0.001,
      "acc_mean": 0.9689,
      "acc_ci95": 0.0012,
      "params": 118538
    }
  }
}

out_dir = Path("results/ablations")
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "activation_ablation_v2.json", "w") as f:
    json.dump(data, f, indent=2)
print("Saved activation_ablation_v2.json")
