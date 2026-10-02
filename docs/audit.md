# NeuraForge Audit Report

**Date of original audit**: Reconstructed from `docs/verification_report.md`
**Date of this document**: 2026-10-02
**Status**: All critical issues resolved. Two running jobs (CNN, Fashion-MNIST ablations) pending.

---

## Background

A verification audit was run against the NeuraForge repository after documentation was found to contain fabricated claims. This document summarizes the findings, fixes applied, and current verified state.

The pre-fix state is captured in [`eval/phase0_asis.json`](../eval/phase0_asis.json). That file was reconstructed after the fact from surviving artifacts (git history, results CSV files, and the verification report). The original pre-fix state was never formally snapshot-ted at the time. A documented gap is preferable to an invented baseline.

---

## Issues Found, By Severity

### 1. FABRICATED: Transformer Training Claim (FIXED)

**Pre-fix**: Documentation claimed the framework "successfully trains" a transformer that produces Shakespearean text. The training script `train_shakespeare.py` existed but had **never been run**. No model weights, no loss curves, no generated text existed anywhere in the repository.

**Fix applied**: The transformer was actually trained. Verified results in `results/shakespeare/`:
- Config: d_model=32, n_layers=1, n_heads=2, d_ff=128, seq_len=32, 14,848 parameters
- 1500 steps, lr=3e-3 with 100-step warmup
- Initial loss 4.1882 (matches ln(65)=4.174 for vocab of 65)
- Best validation loss: 2.1033, perplexity 8.19
- Bigram baseline: 2.4546, perplexity 11.64 — beaten from ~step 250
- Single-batch overfit: loss 4.18 → 1.88 in 50 steps
- Generated text shows learned structure (character names with colons, recognizable English words) but is NOT coherent text
- The larger 4-layer d_model=128 config was NOT trained to convergence

**Evidence**: `results/shakespeare/training_results.json`, `results/shakespeare/training_curves.png`, `results/shakespeare/generated_sample.txt`

### 2. BROKEN: BatchNorm Evaluation Mode (FIXED)

**Pre-fix**: BatchNorm1d and BatchNorm2d in eval mode had forward diffs of ~11.5 and ~12.9 against PyTorch. Root cause: biased variance used for running stats where PyTorch uses unbiased (ddof=1), and eval mode was ignored in the backward pass.

**Post-fix**: Both now pass at <5e-07 against PyTorch.

**Evidence**: `docs/verification_report.md` Phase 4 (pre-fix), post-fix verified in test suite.

### 3. MISSING: MultiHeadAttention Parity Test (FIXED)

**Pre-fix**: No PyTorch parity test existed for MultiHeadAttention at all. The layer's correctness was unverified.

**Post-fix**: Test written and passes at under 2.69e-06 with and without causal masks.

### 4. INVALID: Ablation Comparisons (FIXED)

**Pre-fix**: Ablations used untuned default learning rates for baselines:
- Single seed (std=0.0, zero statistical power)
- Fixed LR for all activations (0.05)
- Fixed LR for optimizers without sweeping
- No validation split for hyperparameter tuning

**Post-fix**: Complete re-run with:
- 7-value LR sweep (1e-4 to 1e-1) on a held-out validation split
- 5 seeds with 95% confidence intervals
- Parameter-matched baselines (PReLU and SwishLearned vs ForageAct)

**Result**: Both custom primitives rejected under fair comparison. ForageAct carries 256 extra parameters and shows no advantage over parameter-matched baselines. NeuroGrad does not beat tuned Adam.

**Evidence**: `results/ablations/activation_ablation_v2.json`, `results/ablations/optimizer_ablation_v2.json`. Old results preserved in `results/activation_ablation/` and `results/optimizer_comparison/` for comparison.

### 5. MISLEADING: "Almost as Fast as PyTorch" Scaling Claim (FIXED)

**Pre-fix**: Claimed 3.96 ms/step vs PyTorch's 3.58 ms/step. This was a measurement artifact — on tiny MLPs, both frameworks dispatch to the same BLAS backend.

**Post-fix**: Replaced with a proper scaling study across MLP, CNN, and Transformer workloads at multiple batch sizes. PyTorch memory column dropped because tracemalloc cannot see C++ allocator activity.

**Evidence**: `results/scaling/scaling_results.json`, `results/scaling/system_info.json`

### 6. FALSE POSITIVE: Autograd reshape/transpose "Failures" (RESOLVED — Not a Bug)

**Pre-fix**: The verification harness reported reshape (1.00e+08) and transpose (1.00e+08) as gradient failures.

**Post-fix**: This was a bug in the verification harness itself, which was modifying aliased views in place. The autograd engine was correct all along. After fixing the harness: reshape 1.18e-10, transpose exactly 0.00e+00.

### 7. BROKEN: Gradio App Predictions (FIXED)

**Pre-fix**: `predict_digit` returned near-uniform probabilities because of a missing Standardizer (training used MNIST mean/std on 0-255 inputs; the app only divided by 255) and Gradio's sketchpad using black-on-white while MNIST expects white-on-black.

**Post-fix**: Fixed. A real MNIST test image with label 6 now predicts 6 at 1.0000.

---

## Pre-Fix vs Post-Fix Summary

| Component | Pre-Fix State | Post-Fix State |
|-----------|---------------|----------------|
| Transformer training | Fabricated claim, never run | Trained, val loss 2.10, perplexity 8.19 |
| BatchNorm eval | fwd diff 11.5–12.9 | fwd diff <5e-07 |
| MultiHeadAttention test | Did not exist | Passes at <2.7e-06 |
| Ablations | 1 seed, no LR sweep, invalid | 5 seeds, 7-LR sweep, 95% CI |
| Scaling claim | Measured BLAS vs BLAS | Full scaling study, 3 workloads |
| Autograd reshape/transpose | Harness bug, falsely reported | Engine correct (1.18e-10, 0.00e+00) |
| Gradio predictions | Near-uniform (broken) | Correct (1.0000 on label 6) |

---

## What Could Not Be Reconstructed

The exact pre-fix snapshot (`eval/phase0_asis.json`) was reconstructed from:
- Git commit history (17 commits available)
- `results/activation_ablation/MNIST/summary.csv` (original single-seed ablation data)
- `results/optimizer_comparison/MNIST/summary.csv` (original single-seed optimizer data)
- `results/performance/benchmark.csv` (original tiny-MLP timing)
- `docs/verification_report.md` (raw terminal output from the audit)

Items that could not be precisely reconstructed:
- The exact README text at the time of the audit (it was modified in-place)
- Any intermediate training states (no checkpoints were saved pre-fix)
- The exact documentation claims (overwritten, not preserved in git)

This is a genuine limitation of reconstructing history after the fact. We document the gap rather than fabricate a "before" snapshot.

---

## Methodology Note

The audit followed a specific discipline:
1. Every claim was checked against actual files in `results/`
2. PyTorch parity was tested at float32 with explicit tolerances
3. Gradient checks used float64 with eps=1e-6
4. Ablations used a validation split for LR tuning (never peeked at test set for hyperparameter selection)
5. Scaling measurements pinned OMP_NUM_THREADS=4, MKL_NUM_THREADS=4, used 1 warmup and 3 repetitions
6. When the harness itself was suspected (reshape/transpose case), the harness was fixed rather than the engine being "fixed" to pass a broken test
