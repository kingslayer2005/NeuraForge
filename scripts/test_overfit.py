import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train_shakespeare import CharTokenizer, get_batch, compute_loss_and_grad
from neuraforge.transformer import DecoderTransformer
from neuraforge.optimizers import Adam

def main():
    # 1. Initialize model
    vocab_size = 65
    model = DecoderTransformer(
        vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=2, d_ff=256, max_seq_len=64, dropout=0.0
    )
    
    # 2. Get a single static batch
    np.random.seed(42)
    B = 4
    T = 64
    x = np.random.randint(0, vocab_size, size=(B, T))
    y = np.random.randint(0, vocab_size, size=(B, T))
    
    # 3. Train
    opt = Adam(model.parameters(), lr=1e-3)
    
    for step in range(50):
        logits = model.forward(x, training=True)
        loss, d_logits = compute_loss_and_grad(logits, y)
        opt.zero_grad()
        model.backward(d_logits)
        opt.step()
        print(f"Step {step}: Loss = {loss:.4f}")

if __name__ == "__main__":
    main()
