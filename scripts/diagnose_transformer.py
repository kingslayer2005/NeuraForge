import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train_shakespeare import (
    CharTokenizer,
    compute_loss_and_grad,
    download_tiny_shakespeare,
    get_batch,
)
from neuraforge.transformer import DecoderTransformer


def compute_bigram_baseline(data, vocab_size):
    # Count transitions
    counts = np.zeros((vocab_size, vocab_size))
    for i in range(len(data)-1):
        counts[data[i], data[i+1]] += 1
    
    # Add Laplace smoothing to avoid log(0)
    counts += 1.0
    probs = counts / counts.sum(axis=1, keepdims=True)
    
    # Compute negative log likelihood on data
    nll = 0
    for i in range(len(data)-1):
        nll -= np.log(probs[data[i], data[i+1]])
    nll /= (len(data)-1)
    
    return nll, np.exp(nll)

def main():
    data_dir = Path("data")
    text = download_tiny_shakespeare(data_dir)
    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)
    
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")
    
    # 1. Uniform Random Loss
    uniform_loss = np.log(vocab_size)
    print(f"Uniform Random Loss: {uniform_loss:.4f} (Perplexity: {np.exp(uniform_loss):.4f})")
    
    # 2. Bigram Baseline
    val_data = data[int(len(data)*0.9):]
    train_data = data[:int(len(data)*0.9)]
    bigram_loss, bigram_perp = compute_bigram_baseline(train_data, vocab_size)
    print(f"Bigram Baseline Train Loss: {bigram_loss:.4f} (Perplexity: {bigram_perp:.4f})")
    bigram_val_loss, bigram_val_perp = compute_bigram_baseline(val_data, vocab_size)
    print(f"Bigram Baseline Val Loss: {bigram_val_loss:.4f} (Perplexity: {bigram_val_perp:.4f})")
    
    # 3. Step 0 Loss of our model
    model = DecoderTransformer(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_seq_len=256
    )
    x, y = get_batch(train_data, batch_size=32, seq_len=128)
    logits = model.forward(x, training=True)
    loss, _ = compute_loss_and_grad(logits, y)
    print(f"Model Step 0 Loss: {loss:.4f} (Perplexity: {np.exp(loss):.4f})")

if __name__ == "__main__":
    main()
