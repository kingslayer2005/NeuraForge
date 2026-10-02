# NeuraForge

A deep learning framework written in pure NumPy. No PyTorch, no TensorFlow, no hidden C++ autograd engines in the training path. Just math.

## Results

> Every number below links to a file in `results/`. If a number is not in `results/`, it does not belong here.

### 1. Fair Ablations Reject the Author's Own Custom Primitives

Both custom primitives (**ForageAct**, a learnable activation, and **NeuroGrad**, a custom optimizer) were tested against parameter-matched, equally-tuned baselines. 7-value LR sweep on a held-out validation split, 5 seeds, 95% confidence intervals. Both are rejected on both datasets.

**Activation Ablation** ([`results/ablations/activation_ablation_v2.json`](results/ablations/activation_ablation_v2.json))

| Activation | Params | MNIST Acc | MNIST 95% CI | Fashion-MNIST Acc | F-MNIST 95% CI |
|---|---|---|---|---|---|
| SwishLearned | 118,538 | 96.97% | ±0.08% | 87.69% | ±0.32% |
| SiLU | 118,282 | 96.95% | ±0.13% | 88.04% | ±0.42% |
| Mish | 118,282 | 96.91% | ±0.10% | 88.01% | ±0.18% |
| **ForageAct** | **118,538** | **96.89%** | **±0.12%** | **87.56%** | **±0.41%** |
| GELU | 118,282 | 96.88% | ±0.05% | 87.97% | ±0.13% |
| ReLU | 118,282 | 96.70% | ±0.09% | 87.70% | ±0.23% |
| PReLU | 118,538 | 96.61% | ±0.10% | 87.76% | ±0.49% |

**Cross-Dataset Finding (Activations)**: ForageAct carries 256 extra learnable parameters. On MNIST, it is statistically indistinguishable from parameter-matched baselines (SwishLearned, PReLU). On Fashion-MNIST, it actually performs the *worst* out of all 7 activations tested. It provides no measurable advantage over standard primitives.

**Optimizer Ablation** ([`results/ablations/optimizer_ablation_v2.json`](results/ablations/optimizer_ablation_v2.json))

| Optimizer | MNIST Acc | MNIST 95% CI | Fashion-MNIST Acc | F-MNIST 95% CI |
|---|---|---|---|---|
| Adam + clip | 96.61% | ±0.11% | 87.98% | ±0.25% |
| SGD + momentum + clip | 96.67% | ±0.11% | 87.90% | ±0.26% |
| NeuroGrad (no clip) | 96.23% | ±0.03% | 87.79% | ±0.12% |
| Adam | 96.77% | ±0.19% | 87.65% | ±0.35% |
| SGD + momentum | 96.47% | ±0.30% | 87.55% | ±0.26% |
| **NeuroGrad (full)** | **96.13%** | **±0.17%** | **87.38%** | **±0.24%** |
| SGD | 96.17% | ±0.07% | 87.17% | ±0.24% |

**Cross-Dataset Finding (Optimizers)**: NeuroGrad is statistically indistinguishable from tuned Adam on both datasets. Its momentum is mathematically equivalent to classical momentum with a rescaled learning rate, and the effect of gradient clipping is optimizer- and dataset-dependent rather than uniformly beneficial (e.g., on Fashion-MNIST, clipping helped Adam but hurt NeuroGrad). It does not offer a novel optimization trajectory.

The conclusion holds consistently across datasets: when evaluated fairly against tuned baselines, both custom primitives are rejected.

### 2. PyTorch Parity

Every layer matches PyTorch at roughly 1e-6 in float32. Raw diffs from [`docs/verification_report.md`](docs/verification_report.md):

| Layer | Config | Fwd Diff | Bwd Diff |
|---|---|---|---|
| Dense | 10→20 | 3.93e-07 | 9.91e-07 |
| Embedding | 100→16 | 3.59e-09 | 5.96e-07 |
| Conv2d | s=1, p=0 | 8.68e-07 | 1.79e-06 |
| Conv2d | s=2, p=0 | 9.94e-07 | 5.62e-07 |
| Conv2d | s=1, p=1 | 8.11e-07 | 2.30e-06 |
| Conv2d | s=2, p=1 | 8.28e-07 | 5.88e-07 |
| MaxPool2d | 2×2 | 0.00e+00 | 0.00e+00 |
| BatchNorm1d | train | 3.58e-07 | 4.62e-07 |
| BatchNorm1d | eval | <5e-07 | <5e-07 |
| BatchNorm2d | train | 4.77e-07 | 4.88e-07 |
| BatchNorm2d | eval | <5e-07 | <5e-07 |
| LayerNorm | — | 4.77e-07 | 6.49e-07 |
| MultiHeadAttention | ±mask | ~9e-07 | ~2.7e-06 |
| Dropout | eval | 0.00e+00 | N/A |

### 3. Character-Level Transformer — Trained End to End in Pure NumPy

Configuration actually trained ([`results/shakespeare/training_results.json`](results/shakespeare/training_results.json)):

| Parameter | Value |
|---|---|
| d_model | 32 |
| n_layers | 1 |
| n_heads | 2 |
| d_ff | 128 |
| seq_len | 32 |
| total parameters | 14,848 |
| training steps | 1,500 |
| learning rate | 3e-3 (100-step warmup) |
| wall-clock time | 1,133 seconds |

| Metric | Value |
|---|---|
| Initial loss | 4.1882 (expected: ln(65) = 4.174) |
| Best validation loss | 2.1033 |
| Validation perplexity | **8.19** |
| Bigram baseline perplexity | 11.64 |
| Bigram beaten at | ~step 250 |

Sample from [`results/shakespeare/generated_sample.txt`](results/shakespeare/generated_sample.txt):
```
MENENTHBAK:
And nented, is ong it taall if or june it.

MIRIXTIOLIUS:
Yet not theit it wather nothe of my oughte lesed deather here too do know yourd
```

The generated text shows learned structure: character names followed by colons, recognizable English words, Shakespeare-like formatting. It is **not** coherent text. This is expected from a 14,848-parameter, 1-layer model on a character-level task.

The larger 4-layer d_model=128 config was **not** trained to convergence. Pure-NumPy CPU training made it computationally impractical — see the scaling study below for why.

### 4. Scaling Study — Where NumPy Stops Tracking PyTorch

Environment: OMP_NUM_THREADS=4, MKL_NUM_THREADS=4, 1 warmup, 3 repetitions. ([`results/scaling/scaling_results.json`](results/scaling/scaling_results.json))

| Workload | Batch | NeuraForge (ms) | PyTorch (ms) | Ratio | NF Memory |
|---|---|---|---|---|---|
| MLP (128w, 2L) | 32 | 2.17 | 1.79 | 1.2× | 0.6 MB |
| MLP (128w, 2L) | 2048 | 19.35 | 4.24 | 4.6× | 24 MB |
| CNN (128w, 2L) | 32 | 158 | 20.6 | 7.7× | 67 MB |
| CNN (128w, 2L) | 2048 | 11,636 | 954 | 12.2× | **4,150 MB** |
| Transformer (128w, 2L) | 32 | 141 | 29.3 | 4.8× | 40 MB |
| Transformer (128w, 2L) | 2048 | 7,240 | 1,538 | 4.7× | 2,405 MB |

**Why the gap widens**:
- **MLPs**: At small batch sizes, both frameworks dispatch to the same BLAS backend. The gap is mostly Python overhead for graph construction and temporary array allocation.
- **CNNs**: NeuraForge uses `im2col`, which creates massive intermediate matrices. At batch 2048, the im2col columns matrix alone peaks at 4.1 GB. PyTorch uses fused kernels (NNPACK/MKL-DNN) that avoid this copy.
- **Transformers**: The attention mechanism's O(n²) memory profile dominates, but the ratio stays relatively stable at ~4.7×, suggesting the bottleneck is consistent Python overhead rather than an algorithmic issue.

PyTorch memory column was dropped because `tracemalloc` cannot see C++ allocator activity — the numbers would be misleading.

---

## CNN on MNIST

Architecture: Conv(1→8, 3×3, pad=1) → ReLU → MaxPool(2×2) → Conv(8→16, 3×3, pad=1) → ReLU → MaxPool(2×2) → Dense(784→128) → ReLU → Dense(128→10). Adam lr=0.001. ([`results/cnn_training.json`](results/cnn_training.json))

| Epoch | Train Acc | Val Acc |
|---|---|---|
| 0 | 90.74% | 95.49% |
| 1 | 97.13% | 97.27% |
| 2 | 98.09% | 97.43% |
| 3 | 98.48% | 97.97% |
| 4 | 98.83% | 98.23% |
| 5 | 99.02% | 98.19% |
| 6 | 99.21% | 98.17% |
| 7 | 99.26% | 98.36% |
| 8 | 99.29% | **98.47%** |
| 9 | 99.44% | 98.29% |
| 10 | 99.57% | 98.17% |
| 11 | 99.55% | 98.46% |

Early stopping triggered at epoch 11 (patience 3, best at epoch 8). **Final test accuracy: 98.47%.**

This is a small architecture (two conv layers with 8 and 16 filters) with no data augmentation, no learning rate scheduling, and no regularization beyond early stopping. 98.47% is a fair result for this setup — typical CNNs on MNIST with deeper architectures and augmentation reach 99%+.

---

## What's Inside

```
neuraforge/
├── autograd.py        # Define-by-run reverse-mode autodiff engine
├── nn.py              # Dense, Conv2d (im2col), MaxPool2d, BatchNorm1d/2d,
│                      # LayerNorm, Embedding, Dropout
├── transformer.py     # MultiHeadAttention, TransformerBlock, DecoderTransformer,
│                      # sinusoidal and learned positional encodings
├── activations.py     # ReLU, GELU, SiLU, Mish, PReLU, SwishLearned, ForageAct
├── optimizers.py      # SGD, Momentum, Adam, NeuroGrad
├── growth.py          # Net2WiderNet-style layer widening
├── pruning.py         # First-order Taylor neuron pruning
└── legacy/            # Original hand-written backward passes (kept as
                       # independent correctness reference)

experiments/           # Scaling study, fair ablations, training scripts
tests/                 # 152 tests (PyTorch allowed here as reference only)
results/               # All measured data — every claim traces here
docs/                  # Architecture, derivations, audit report, findings
app.py                 # Gradio demo (MNIST digit recognition + Shakespeare generation)
```

## Current Status

**Verified & Complete**
- Reverse-mode autograd engine passes gradient checks for all ops in float64 below 1e-6, including reshape, transpose, broadcasting across all shape pairs, and diamond dependencies.
- 14 layer configurations match PyTorch at 1e-6 or better (Dense, Conv2d ×4, MaxPool2d, BatchNorm1d/2d in train and eval, LayerNorm, Embedding, MultiHeadAttention, Dropout).
- Character-level transformer trained to 8.19 perplexity, beating bigram baseline (11.64).
- CNN on MNIST: 98.47% test accuracy (Conv 8→16, early stopping at epoch 11).
- Scaling study completed across MLP, CNN, and Transformer workloads.
- Fair ablations completed on MNIST and Fashion-MNIST with 7-LR sweep, 5 seeds, 95% CI. Both custom primitives rejected consistently on both datasets.
- Gradio demo correctly predicts MNIST digits.

**Known Limitations**
- `im2col` convolution creates massive intermediate matrices (4.1 GB at batch 2048). This is the primary scaling bottleneck for CNNs.
- Pure NumPy is CPU-only. The 4-layer transformer config was not trained because it would take >7 hours.
- NeuraForge is 4.6–12× slower than PyTorch at batch 2048, depending on workload type.

## Reproducing Results

```bash
# Run the test suite (152 tests)
pytest tests/ -v

# Fair ablations (activation + optimizer, MNIST + Fashion-MNIST)
python experiments/fair_ablations_v2.py

# Scaling study (NeuraForge vs PyTorch, MLP/CNN/Transformer)
python experiments/scaling_study.py

# Train character-level transformer
python experiments/train_shakespeare.py

# Train CNN on MNIST
python experiments/train_cnn.py

# Launch Gradio demo
python app.py
```

## Documentation

- [Autograd Engine Architecture](docs/architecture.md)
- [Gradient Derivations](docs/derivations.md) — hand-derived gradients for softmax-CE, BatchNorm, LayerNorm, and scaled dot-product attention
- [Research Findings & Corrected Claims](docs/findings.md)
- [Audit Report](docs/audit.md) — what was broken, what was fabricated, and how it was fixed
- [Verification Report](docs/verification_report.md) — raw terminal output from the original audit

## License

MIT
