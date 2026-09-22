import pandas as pd
from pathlib import Path
import os
import re

def main():
    root_dir = Path(__file__).parent.parent
    results_dir = root_dir / "results"
    readme_path = root_dir / "README.md"
    
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()

    # 1. Performance
    perf_csv = results_dir / "performance" / "benchmark.csv"
    if perf_csv.exists():
        perf_df = pd.read_csv(perf_csv)
        perf_table = "| Framework | Precision | Time (ms/step) |\n|-----------|-----------|----------------|\n"
        for _, row in perf_df.iterrows():
            fw = f"**{row['Framework']}**" if "NeuraForge" in row['Framework'] else row['Framework']
            pr = f"**{row['Precision']}**" if "NeuraForge" in row['Framework'] else row['Precision']
            time_str = f"**{row['Time (ms/iter)']:.2f} ms**" if "NeuraForge" in row['Framework'] else f"{row['Time (ms/iter)']:.2f} ms"
            perf_table += f"| {fw} | {pr} | {time_str} |\n"
        
        # Replace in README
        readme_content = re.sub(
            r"\| Framework \| Precision \| Time \(ms/step\) \|\n\|[-]+\|[-]+\|[-]+\|\n(?:\|.*?\|\n)*",
            perf_table,
            readme_content
        )

    # 2. Activation Ablation (MNIST)
    act_csv = results_dir / "activation_ablation" / "MNIST" / "summary.csv"
    if act_csv.exists():
        act_df = pd.read_csv(act_csv)
        act_table = "| Activation | Best LR | Test Accuracy |\n|------------|---------|---------------|\n"
        for _, row in act_df.iterrows():
            act = row['Activation']
            lr = row['Best_LR']
            acc = row['Test_Acc_Mean']
            std = row['Test_Acc_Std']
            
            acc_str = f"{acc*100:.2f}% ± {std*100:.2f}%"
            if "ForageAct" in act:
                act = f"**{act}**"
                lr = f"**{lr}**"
                acc_str = f"**{acc_str}**"
            act_table += f"| {act} | {lr} | {acc_str} |\n"
            
        readme_content = re.sub(
            r"\| Activation \| Best LR \| Test Accuracy \|\n\|[-]+\|[-]+\|[-]+\|\n(?:\|.*?\|\n)*",
            act_table,
            readme_content
        )

    # 3. Optimizer Comparison (MNIST)
    opt_csv = results_dir / "optimizer_comparison" / "MNIST" / "summary.csv"
    if opt_csv.exists():
        opt_df = pd.read_csv(opt_csv)
        opt_table = "| Optimizer | Best LR | Test Accuracy |\n|-----------|---------|---------------|\n"
        for _, row in opt_df.iterrows():
            opt = row['Optimizer']
            lr = row['Best_LR']
            acc = row['Test_Acc_Mean']
            std = row['Test_Acc_Std']
            
            acc_str = f"{acc*100:.2f}% ± {std*100:.2f}%"
            if "NeuroGrad" in opt:
                opt = f"**{opt}**"
                lr = f"**{lr}**"
                acc_str = f"**{acc_str}**"
            opt_table += f"| {opt} | {lr} | {acc_str} |\n"
            
        readme_content = re.sub(
            r"\| Optimizer \| Best LR \| Test Accuracy \|\n\|[-]+\|[-]+\|[-]+\|\n(?:\|.*?\|\n)*",
            opt_table,
            readme_content
        )
        
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
        
    print("README updated successfully.")

if __name__ == "__main__":
    main()
