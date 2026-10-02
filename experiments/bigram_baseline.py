import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.train_shakespeare import download_tiny_shakespeare, CharTokenizer

def bigram_baseline():
    root = Path(__file__).parent.parent
    data_dir = root / "data"
    text = download_tiny_shakespeare(data_dir)
    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)
    
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    vocab_size = tokenizer.vocab_size
    counts = np.ones((vocab_size, vocab_size)) # Add-1 smoothing
    
    # Count bigrams
    for i in range(len(train_data) - 1):
        counts[train_data[i], train_data[i+1]] += 1
        
    probs = counts / np.sum(counts, axis=1, keepdims=True)
    
    # Eval on val data
    log_prob_sum = 0.0
    for i in range(len(val_data) - 1):
        log_prob_sum += np.log(probs[val_data[i], val_data[i+1]])
        
    avg_nll = -log_prob_sum / (len(val_data) - 1)
    perplexity = np.exp(avg_nll)
    
    print(f"Bigram Validation NLL: {avg_nll:.4f}")
    print(f"Bigram Validation Perplexity: {perplexity:.4f}")

if __name__ == "__main__":
    bigram_baseline()
