import json
import time

print("Waiting for optimizer ablations to finish...")
while True:
    try:
        with open('results/ablations/optimizer_ablation_v2.json', 'r') as f:
            d = json.load(f)
        fm = d.get('Fashion-MNIST', {})
        if len(fm) == 7:
            print("Fashion-MNIST optimizers completed: 7/7")
            for k, v in fm.items():
                acc = v["acc_mean"] * 100
                ci = v["acc_ci95"] * 100
                print(f"  {k}: {acc:.2f}% +/- {ci:.2f}%")
            break
    except Exception:
        pass
    time.sleep(10)
