# NeuraForge — Progress Tracker

## Phase 1: Package Structure — STATUS: DONE

Refactor v1–v4 into `neuraforge/` package. Key components:
- `layers.py` (Dense, Dropout)
- `activations.py` (ReLU, LeakyReLU, Tanh, Sigmoid, SiLU, GELU, ForageAct)
- `losses.py` (SoftmaxCrossEntropy, MSE, BCEWithLogits)
- `optimizers.py` (SGD, MomentumSGD, Adam, NeuroGrad)
- `model.py` (Sequential with Parameter registry)
- `data.py` (mini-batch loader, splits, standardisation)
- `train.py` (fit/evaluate with early stopping + best-weight restore)
- `io.py` (save/load weights + architecture config as .npz)
- `seed.py` (seed_everything)
- Move v1–v4 to `legacy/`
- `pyproject.toml`, `requirements.txt`, `.gitignore`
- Rename "NeuroForage" → "NeuraForge" everywhere

### Corrections applied
- Parameter class with .name, .data, .grad (not dicts)
- Scalar alpha stored as np.array of shape (1,)
- io.py saves architecture config alongside weights
- MPLBACKEND=Agg, save plots to files, never plt.show()
- fit() reports sample-weighted average loss per epoch
- Best-weight restore on early stopping

---

## Phase 2: Prove Correctness — STATUS: DONE

- `tests/test_gradcheck.py`: central finite-difference gradient checks for every layer, activation (including d/d-alpha for ForageAct), and loss, in float64, relative error < 1e-6.
- `tests/test_torch_parity.py`: load identical weights into NeuraForge and an equivalent PyTorch model (with ForageAct written as a torch module); forward outputs and every gradient must match within 1e-6 for a 3-layer MLP.
- Optimizer tests on a simple convex quadratic (each optimizer must converge to the known minimum).
- If torch does not install on Python 3.14, create a separate dev environment with uv (`py -m pip install uv`, then `uv venv --python 3.12 .venv-dev`) used only for tests and benchmarks.

---

## Phase 3: Real Benchmarks — STATUS: IN PROGRESS

- Datasets: two-moons and spirals (for decision-boundary plots), MNIST and Fashion-MNIST (load via sklearn fetch_openml, cache as .npz in data/, gitignored).
- Experiments in `experiments/`, results as CSV/JSON in `results/`, 5 seeds, mean ± std, the same epoch budget for every run, learning rate tuned per optimizer on validation only:
  a. Activation ablation: ReLU, GELU, SiLU, ForageAct (alpha fixed at 0.1), ForageAct (learnable scalar alpha), ForageAct (learnable per-neuron alpha). Also log how alpha evolves during training.
  b. Optimizer comparison: SGD, Momentum, Adam, NeuroGrad, NeuroGrad without clipping, and per-tensor vs global-norm clipping.
- All model selection happens on validation. Report test accuracy only for the final chosen configurations.

---

## Phase 4: Adaptive Architecture — SPEC (from original prompt)

- Neuron growth: Net2WiderNet-style function-preserving widening of a hidden layer when validation loss plateaus. Add a test proving the network's outputs are unchanged immediately after widening (difference < 1e-8).
- Structured pruning: remove hidden neurons by importance (weight-norm score and first-order Taylor score), then fine-tune. Plot accuracy vs parameter count.
- Compare fixed-small, fixed-large, grow-from-small and prune-from-large at matched parameter counts, 5 seeds each.

---

## Phase 5: Performance — SPEC (from original prompt)

- float32 path. Benchmark epoch time against PyTorch CPU on the same MLP and report it honestly (NumPy will likely be slower; explain why).
- Optional stretch, only if approved: Conv2D via im2col, with gradient checks.

---

## Phase 6: Public Demo — SPEC (from original prompt)

- A Gradio app deployable free on Hugging Face Spaces: draw a digit and get the NumPy-only model's prediction with class probabilities, plus a tab showing training curves and the experiment tables read from results/. Include step-by-step deployment instructions.
- User will push to HF themselves. Provide exact commands, never ask for tokens.

---

## Phase 7: Docs and Polish — SPEC (from original prompt)

- README: what and why, architecture diagram (Mermaid), results tables generated from results/, one command to reproduce each experiment, limitations.
- docs/MATH.md: derivation of every backward pass (Dense, each activation including ForageAct and d/d-alpha, softmax cross-entropy) and every optimizer update, in LaTeX, matching the code line for line.
- docs/EXPLAINED.md: plain-language walkthrough of the framework plus 10 likely interview questions with answers grounded in this code.
- GitHub Actions running ruff + pytest on every push (test on Python 3.11 and 3.14).
- Finally, propose 3 resume bullets using only numbers from results/.

---

## Key Ground Rules (preserved)

1. Honesty over impressive numbers. Every number in README/docs/resume must come from a script actually run, saved under results/.
2. Library uses only NumPy (+ matplotlib for plots). PyTorch allowed ONLY in tests and benchmarks.
3. Code style: explicit, intern-level Python. Comment on every meaningful line, docstrings, type hints.
4. Must run on laptop, CPU only. Fixed random seeds everywhere. Pin dependency versions.
5. MPLBACKEND=Agg, save plots, never plt.show().
6. EMA momentum note: without clipping, NeuroGrad is identical to classical momentum with lr*(1-beta). Document in README and MATH.md.
7. Never fake progress: no hardcoded numbers, no stubs presented as finished, no skipped tests.
