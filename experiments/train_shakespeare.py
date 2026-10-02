"""
train_shakespeare.py — Train a character-level transformer on tiny-shakespeare.

Trains a DecoderTransformer to predict the next character given a context
window. Uses NeuraForge's pure-NumPy autograd engine — no PyTorch.

Usage:
    python experiments/train_shakespeare.py

The script will:
    1. Download tiny-shakespeare (~1.1MB) if not cached
    2. Build a character-level tokenizer
    3. Train a decoder-only transformer
    4. Generate sample text every N steps
    5. Save results to results/shakespeare/

Expected runtime: ~30-60 minutes on CPU for a small model.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from neuraforge.optimizers import Adam
from neuraforge.seed import seed_everything
from neuraforge.transformer import DecoderTransformer


# ===================================================================
# DATASET
# ===================================================================
def download_tiny_shakespeare(data_dir: Path) -> str:
    """Download tiny-shakespeare text file if not cached."""
    cache_path = data_dir / "tiny_shakespeare.txt"
    if cache_path.exists():
        print(f"Loading tiny-shakespeare from cache ({cache_path})")
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    print("Downloading tiny-shakespeare...")
    import urllib.request
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, cache_path)

    with open(cache_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Downloaded {len(text)} characters")
    return text


class CharTokenizer:
    """Character-level tokenizer."""

    def __init__(self, text: str):
        # Build vocabulary from the text
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.char_to_idx[c] for c in text], dtype=np.int64)

    def decode(self, tokens: np.ndarray) -> str:
        return "".join(self.idx_to_char[int(t)] for t in tokens)


# ===================================================================
# TRAINING LOOP
# ===================================================================
def get_batch(data: np.ndarray, batch_size: int, seq_len: int):
    """Get a random batch of (input, target) sequences.

    Targets are the inputs shifted right by 1 position.
    """
    max_start = len(data) - seq_len - 1
    starts = np.random.randint(0, max_start, size=batch_size)

    x = np.stack([data[s:s + seq_len] for s in starts])
    y = np.stack([data[s + 1:s + seq_len + 1] for s in starts])
    return x, y


def compute_loss_and_grad(logits: np.ndarray, targets: np.ndarray):
    """Compute softmax cross-entropy loss and return gradient w.r.t. logits.

    Parameters
    ----------
    logits : shape (B, T, V)
    targets : shape (B, T), integer token IDs

    Returns
    -------
    loss : float
    d_logits : shape (B, T, V)
    """
    B, T, V = logits.shape

    # One-hot encode targets
    targets_flat = targets.reshape(-1)
    targets_onehot = np.zeros((B * T, V), dtype=logits.dtype)
    targets_onehot[np.arange(B * T), targets_flat] = 1.0

    # Softmax
    logits_2d = logits.reshape(-1, V)
    shifted = logits_2d - np.max(logits_2d, axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)

    # Cross-entropy loss
    loss = -np.sum(targets_onehot * np.log(probs + 1e-12)) / (B * T)

    # Gradient: (p - y) / (B * T)
    d_logits = ((probs - targets_onehot) / (B * T)).reshape(B, T, V)

    return loss, d_logits


def train():
    seed_everything(42)

    # Paths
    root = Path(__file__).parent.parent
    data_dir = root / "data"
    out_dir = root / "results" / "shakespeare"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    text = download_tiny_shakespeare(data_dir)
    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Dataset size: {len(data)} tokens")

    # Train/val split (90/10)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Model configuration — extremely small for fast verification
    d_model = 32
    n_heads = 2
    n_layers = 1
    d_ff = 128
    max_seq_len = 32
    dropout = 0.0

    batch_size = 32
    seq_len = 32
    learning_rate = 3e-3 # even higher LR for faster convergence
    max_steps = 1500
    warmup_steps = 100
    eval_interval = 100
    eval_steps = 20
    generate_interval = 500

    print(f"\nModel: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, d_ff={d_ff}")

    model = DecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
        tied_embeddings=True,
    )

    params = model.parameters()
    n_params = sum(p.data.size for p in params)
    print(f"Total parameters: {n_params:,}")

    opt = Adam(params, lr=learning_rate)

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    start_time = time.time()

    for step in range(1, max_steps + 1):
        # Warmup
        if step <= warmup_steps:
            opt.lr = learning_rate * (step / warmup_steps)
        else:
            opt.lr = learning_rate
            
        # Get batch
        x_batch, y_batch = get_batch(train_data, batch_size, seq_len)

        # Forward
        logits = model.forward(x_batch, training=True)
        loss, d_logits = compute_loss_and_grad(logits, y_batch)

        # Backward
        opt.zero_grad()
        model.backward(d_logits)

        # Gradient clipping (global norm)
        max_grad_norm = 1.0
        total_norm = 0.0
        for p in params:
            if p.grad is not None:
                total_norm += np.sum(p.grad ** 2)
        total_norm = np.sqrt(total_norm)
        if total_norm > max_grad_norm:
            clip_factor = max_grad_norm / total_norm
            for p in params:
                if p.grad is not None:
                    p.grad *= clip_factor

        opt.step()
        train_losses.append(loss)

        if step % 50 == 0:
            elapsed = time.time() - start_time
            ms_per_step = elapsed / step * 1000
            print(f"Step {step}/{max_steps} | loss: {loss:.4f} | "
                  f"{ms_per_step:.0f} ms/step | "
                  f"grad_norm: {total_norm:.3f}")

        # Evaluation
        if step % eval_interval == 0:
            val_loss_sum = 0.0
            for _ in range(eval_steps):
                x_val, y_val = get_batch(val_data, batch_size, seq_len)
                logits_val = model.forward(x_val, training=False)
                vl, _ = compute_loss_and_grad(logits_val, y_val)
                val_loss_sum += vl
            val_loss = val_loss_sum / eval_steps
            val_losses.append((step, val_loss))
            print(f"  [eval] step {step} | val_loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model params
                save_dict = {p.name: p.data for p in params}
                np.savez(out_dir / "best_model.npz", **save_dict)
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    print(f"Early stopping triggered at step {step}")
                    break

        # Generate sample text
        if step % generate_interval == 0:
            prompt = "\nHAMLET:\n"
            prompt_tokens = tokenizer.encode(prompt)
            start_tokens = prompt_tokens[np.newaxis, :]

            generated = model.generate(start_tokens, max_new_tokens=200, temperature=0.8)
            generated_text = tokenizer.decode(generated[0])
            print(f"\n--- Generated text (step {step}) ---")
            print(generated_text[:300])
            print("--- end ---\n")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Save final results
    import json
    results = {
        "total_params": n_params,
        "total_time_seconds": total_time,
        "best_val_loss": best_val_loss,
        "final_train_loss": float(train_losses[-1]),
        "config": {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "max_seq_len": max_seq_len,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "learning_rate": learning_rate,
            "max_steps": max_steps,
        }
    }
    with open(out_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save training curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Training loss (smoothed)
        window = 50
        if len(train_losses) > window:
            smoothed = np.convolve(train_losses, np.ones(window)/window, mode='valid')
            ax1.plot(smoothed, linewidth=1.5)
        else:
            ax1.plot(train_losses, linewidth=1.5)
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)

        # Validation loss
        if val_losses:
            steps, losses = zip(*val_losses)
            ax2.plot(steps, losses, 'o-', linewidth=2)
            ax2.set_title("Validation Loss")
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Loss")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / "training_curves.png", dpi=150)
        plt.close()
        print(f"Saved training curves to {out_dir / 'training_curves.png'}")
    except ImportError:
        pass

    # Final generation
    prompt = "\nTo be, or not to be, that is the question:\n"
    prompt_tokens = tokenizer.encode(prompt)
    start_tokens = prompt_tokens[np.newaxis, :]
    generated = model.generate(start_tokens, max_new_tokens=500, temperature=0.8)
    generated_text = tokenizer.decode(generated[0])
    print(f"\n{'='*60}")
    print("FINAL GENERATED TEXT:")
    print('='*60)
    print(generated_text[:600])

    with open(out_dir / "generated_sample.txt", "w", encoding="utf-8") as f:
        f.write(generated_text)


if __name__ == "__main__":
    train()
