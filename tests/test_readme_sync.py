import pandas as pd
from pathlib import Path
import re

def test_readme_matches_results():
    root_dir = Path(__file__).parent.parent
    results_dir = root_dir / "results"
    readme_path = root_dir / "README.md"
    
    if not readme_path.exists():
        return
        
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()

    # Check Performance
    perf_csv = results_dir / "performance" / "benchmark.csv"
    if perf_csv.exists():
        perf_df = pd.read_csv(perf_csv)
        for _, row in perf_df.iterrows():
            time_str = f"{row['Time (ms/iter)']:.2f} ms"
            assert time_str in readme_content, f"README is out of sync with performance results. Missing {time_str}"

    # Check Activation Ablation
    act_csv = results_dir / "activation_ablation" / "MNIST" / "summary.csv"
    if act_csv.exists():
        act_df = pd.read_csv(act_csv)
        for _, row in act_df.iterrows():
            acc_str = f"{row['Test_Acc_Mean']*100:.2f}% ± {row['Test_Acc_Std']*100:.2f}%"
            assert acc_str in readme_content, f"README is out of sync with activation ablation results. Missing {acc_str}"

    # Check Optimizer Comparison
    opt_csv = results_dir / "optimizer_comparison" / "MNIST" / "summary.csv"
    if opt_csv.exists():
        opt_df = pd.read_csv(opt_csv)
        for _, row in opt_df.iterrows():
            acc_str = f"{row['Test_Acc_Mean']*100:.2f}% ± {row['Test_Acc_Std']*100:.2f}%"
            assert acc_str in readme_content, f"README is out of sync with optimizer comparison results. Missing {acc_str}"
