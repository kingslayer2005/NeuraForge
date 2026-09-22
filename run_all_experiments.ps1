$env:PYTHONPATH="."
Write-Host "Running ablation on MNIST..."
.\.venv-dev\Scripts\python.exe experiments/ablation_activations.py --dataset MNIST --seeds 5 --epochs 15
Write-Host "Running ablation on Fashion-MNIST..."
.\.venv-dev\Scripts\python.exe experiments/ablation_activations.py --dataset Fashion-MNIST --seeds 5 --epochs 15
Write-Host "Running optimizers on MNIST..."
.\.venv-dev\Scripts\python.exe experiments/compare_optimizers.py --dataset MNIST --seeds 5 --epochs 15
Write-Host "Running optimizers on Fashion-MNIST..."
.\.venv-dev\Scripts\python.exe experiments/compare_optimizers.py --dataset Fashion-MNIST --seeds 5 --epochs 15
Write-Host "Running growth/pruning on MNIST..."
.\.venv-dev\Scripts\python.exe experiments/benchmark_growth_pruning.py --seeds 5 --epochs 15
Write-Host "All experiments completed!"
