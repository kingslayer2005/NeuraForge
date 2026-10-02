import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train_shakespeare import CharTokenizer, download_tiny_shakespeare
from neuraforge.transformer import DecoderTransformer


def get_loss(model, text, tokenizer, seq_len=128):
    # Evaluate a few batches to estimate loss
    from neuraforge.autograd import Tensor, softmax_cross_entropy
    np.random.seed(42)
    
    losses = []
    for _ in range(10):
        start_idx = np.random.randint(0, len(text) - seq_len - 1)
        chunk = text[start_idx : start_idx + seq_len + 1]
        tokens = tokenizer.encode(chunk)
        
        X = np.array(tokens[:-1])[None, :]
        Y = np.array(tokens[1:])[None, :]
        
        logits = model.forward(X, training=False)
        logits_flat = logits.reshape(-1, tokenizer.vocab_size)
        Y_flat = Y.reshape(-1)
        
        # one-hot Y
        Y_oh = np.zeros((Y_flat.size, tokenizer.vocab_size))
        Y_oh[np.arange(Y_flat.size), Y_flat] = 1.0
        
        logits_t = Tensor(logits_flat)
        loss = softmax_cross_entropy(logits_t, Y_oh).data
        losses.append(loss)
    
    mean_loss = np.mean(losses)
    return mean_loss, np.exp(mean_loss)

def generate(model, tokenizer, max_new_tokens=100):
    prompt = "O Romeo, Romeo"
    tokens = list(tokenizer.encode(prompt))
    model._eval_mode = True # Just to be safe, though our forward takes training=False
    
    for _ in range(max_new_tokens):
        x = np.array(tokens)[None, :]
        logits = model.forward(x, training=False)
        next_logit = logits[0, -1, :]
        
        # basic sample
        probs = np.exp(next_logit - np.max(next_logit))
        probs /= np.sum(probs)
        next_token = np.random.choice(tokenizer.vocab_size, p=probs)
        tokens.append(next_token)
        
    return tokenizer.decode(tokens)

def main():
    text = download_tiny_shakespeare(Path("data"))
    tokenizer = CharTokenizer(text)
    
    model = DecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_seq_len=256
    )
    from neuraforge.io import set_weights
    data = np.load("results/shakespeare/best_model.npz", allow_pickle=True)
    weights = {k: v for k, v in data.items() if k != "__config__"}
    set_weights(model, weights)
    
    val_text = text[int(0.9 * len(text)):]
    loss, perp = get_loss(model, val_text, tokenizer)
    gen_text = generate(model, tokenizer)
    
    print(f"B1 Final Loss: {loss:.4f}")
    print(f"B1 Final Perplexity: {perp:.4f}")
    print(f"B1 Generated Text:\n{gen_text}")

if __name__ == "__main__":
    main()
