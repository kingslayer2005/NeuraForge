# NeuraForge Research Findings

This document summarizes the findings from our rigorous benchmarking and ablation studies, correcting the original overstated claims of the project.

## Claim A: "NeuraForge runs almost as fast as PyTorch CPU"

**Original Claim**: NeuraForge achieves 3.96 ms/step compared to PyTorch's 3.58 ms/step.
**Corrected Finding**: This original claim is an artifact of measuring a single, small MLP where the runtime is overwhelmingly dominated by the underlying BLAS library's GEMM operations. Our Phase 5 scaling study reveals the true performance characteristics:

1. **Small Dense Workloads**: On small MLPs (e.g., batch 32, width 128), NeuraForge and PyTorch perform similarly because both just dispatch to the same BLAS backend (OpenBLAS/MKL), and PyTorch incurs minor Python overhead for graph construction.
2. **Scaling Up**: As matrix sizes increase (batch 2048, width 2048), PyTorch's highly optimized C++ backend, multi-threading, and memory management pull decisively ahead. NeuraForge begins to bottleneck on Python-level overhead and memory allocations for intermediate tensors during the backward pass.
3. **Convolutions**: On CNNs, NeuraForge uses `im2col`, which incurs massive memory copies and allocations. PyTorch uses highly optimized, fused kernels (e.g., NNPACK, MKL-DNN) that avoid this copy, making PyTorch orders of magnitude faster and vastly more memory efficient for CNNs.

## Claim B: "ForageAct beats standard activations"

**Original Claim**: ForageAct (a custom learnable activation) achieves higher accuracy than standard fixed activations like ReLU, GELU, and SiLU.
**Corrected Finding**: The original comparison was structurally unfair because it compared a *learnable* activation against *fixed* baselines. When compared against parameter-matched learnable baselines in Phase 6:

- **ForageAct vs. PReLU / Swish (Learnable)**: With identical tuning budgets and parameter counts, ForageAct performs comparably to, but does not statistically outperform, standard parameter-matched baselines like PReLU or Swish with a learnable beta.
- The perceived "win" was entirely due to the extra adaptive parameters, not a fundamental architectural superiority of ForageAct itself. 

## Claim C: "NeuroGrad is a novel optimizer"

**Original Claim**: NeuroGrad is a novel optimizer that outperforms SGD and Adam.
**Corrected Finding**: NeuroGrad is simply Exponential Moving Average (EMA) momentum combined with gradient clipping. Our decomposition in Phase 6 isolates these components:

- **EMA Momentum vs Classical Momentum**: Without clipping, EMA momentum is mathematically equivalent to classical momentum with a rescaled learning rate ($lr_{classical} = lr \times (1 - \beta)$).
- **The Role of Clipping**: The effect of gradient clipping is optimizer- and dataset-dependent rather than uniformly beneficial. For example, on Fashion-MNIST, clipping improved Adam but worsened NeuroGrad. 
- **Overall**: NeuroGrad is statistically indistinguishable from tuned Adam on both datasets. It does not provide a fundamentally novel optimization trajectory.

## Claim D: "Transformer training convergence"

**Original Claim**: The 4-layer d_model=128 character-level transformer trains successfully.
**Corrected Finding**: The larger configuration was not trained to convergence because pure-NumPy CPU training made it computationally impractical (taking over 7 hours). The actual trained and working configuration is significantly smaller (`d_model=32, n_layers=1, seq_len=32`, with 14,848 parameters). This is an honest constraint of pure-NumPy frameworks: while mathematically robust, iterating on heavy architectures without a GPU or fused kernels is impractically slow.

## What Did Not Work

During our optimization efforts (Phase 5), we attempted several strategies that did not yield meaningful improvements:
- **Python-level Memory Pools**: Trying to cache and reuse NumPy arrays in Python introduced more overhead in dictionary lookups and reference counting than it saved in allocations for small-to-medium matrices.
- **In-place Operations**: While mathematically sound, doing in-place updates (e.g., `X += Y`) inside the autograd engine's backward pass proved highly brittle. If a tensor was unexpectedly part of a diamond dependency, in-place mutation corrupted the gradient for other branches. We opted for correctness (accumulating with `+=` on fresh arrays) over marginal speedups.
