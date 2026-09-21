# NeuraForge

## Neural Network From Scratch (NumPy Only)

NeuraForge is a neural network framework implemented entirely from first principles using pure NumPy.

No PyTorch.  
No TensorFlow.  
No autograd.

The goal of this project is to deeply understand and extend the mathematical foundations of neural networks by building every component manually.

---

## 🔍 Project Objective

Most deep learning implementations abstract away the core mathematics.

NeuraForge focuses on:

- Manual forward propagation  
- Manual backpropagation  
- Matrix-based gradient computation  
- Custom training loop implementation  
- Full architectural control  

This project is built as a foundation for developing an adaptive, self-evolving neural architecture.

---

## 🧠 Version History

The original standalone scripts (`neuroforage_v1.py` to `v4.py`) that explored the base mechanics are preserved unchanged in the `legacy/` directory for historical context. 

The current active framework lives in the `neuraforge/` Python package.

---

## 🏗 Architecture

Input → Dense → Activation → ... → Dense → Softmax + Cross Entropy Loss  

Every forward and backward step is explicitly implemented using pure NumPy matrix operations.

---

## 📈 Honest Extensions

NeuraForge includes two custom extensions, which we benchmark rigorously against standard baselines (see `results/`):

1. **ForageAct**: `f(z) = z·σ(z) + α·tanh(z)`
   - When α = 0, this is exactly the SiLU/Swish activation function (Ramachandran et al., 2017).
   - The α·tanh(z) term is a learnable extension (with fixed, scalar, or per-neuron modes) designed to add a sign-sensitive bias.
2. **NeuroGrad**:
   - An optimizer using Exponential Moving Average (EMA) momentum (`v = β·v + (1−β)·g`) and gradient-norm clipping (Pascanu et al., 2013).
   - *Note on EMA vs Classical Momentum*: Without clipping, EMA momentum is mathematically identical to classical momentum (`v = β·v + g`) operating at an effective learning rate of `lr_effective = lr * (1 - β)`. Any performance differences observed between NeuroGrad and standard Momentum SGD in our benchmarks therefore stem entirely from the gradient clipping mechanism or the separately tuned learning rates.

---

## 🛠 Tech Stack

- Python >= 3.10
- NumPy
- Matplotlib

---

## 🚀 Roadmap

- Phase 1: Package Structure (done)
- Phase 2: Prove Correctness (gradient checking & Torch parity)
- Phase 3: Real Benchmarks (Activation ablation & Optimizer comparison)
- Phase 4: Adaptive Architecture (Neuron growth & Pruning)
- Phase 5: Performance (float32 benchmarking)
- Phase 6: Public Demo (Gradio deployment)
- Phase 7: Docs & Polish

---

## 📌 Author

Aarush Gupta

