import gradio as gr
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))

from neuraforge.model import Sequential
from neuraforge.layers import Dense
from neuraforge.activations import ReLU
from neuraforge.io import load_model
from neuraforge.transformer import DecoderTransformer
from experiments.train_shakespeare import CharTokenizer, download_tiny_shakespeare

# Load MNIST model
model_path = Path(__file__).parent / "results" / "demo_model.npz"
if model_path.exists():
    model = load_model(str(model_path))
else:
    model = Sequential(Dense(784, 128), ReLU(), Dense(128, 128), ReLU(), Dense(128, 10))

# Load Transformer model and Tokenizer
transformer_path = Path(__file__).parent / "results" / "shakespeare" / "best_model.npz"
text_cache = Path(__file__).parent / "data" / "tiny_shakespeare.txt"

tokenizer = None
transformer_model = None

if text_cache.exists():
    with open(text_cache, "r", encoding="utf-8") as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    
    if transformer_path.exists():
        # Initialize the same model as in train_shakespeare
        transformer_model = DecoderTransformer(
            vocab_size=tokenizer.vocab_size, d_model=32, n_heads=2,
            n_layers=1, d_ff=128, max_seq_len=32
        )
        from neuraforge.io import set_weights
        weights = np.load(transformer_path, allow_pickle=True)
        weights_dict = {k: v for k, v in weights.items() if k != "__config__"}
        set_weights(transformer_model, weights_dict)

def predict_digit(image):
    if image is None: return {str(i): 0.0 for i in range(10)}
    if isinstance(image, dict) and 'composite' in image: img_array = image['composite']
    else: img_array = image
        
    from PIL import Image
    img_pil = Image.fromarray(img_array).convert('L').resize((28, 28))
    img = np.array(img_pil).astype(np.float64)
    
    # Gradio sketchpad is typically black strokes on white canvas.
    # MNIST is white strokes on black canvas.
    # We invert if the background is mostly white
    if img.mean() > 127:
        img = 255.0 - img
        
    # Standardize using MNIST train set mean and std (in 0-255 scale)
    MNIST_MEAN = 33.3184
    MNIST_STD = 78.5675
    img = (img - MNIST_MEAN) / MNIST_STD
    
    X = img.reshape(1, 784)
    logits = model.forward(X, training=False)
    
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return {str(i): float(probs[0][i]) for i in range(10)}

def generate_text(prompt, max_tokens, temperature):
    if transformer_model is None or tokenizer is None:
        return "Model not found. Run train_shakespeare.py first.", None

    prompt_tokens = tokenizer.encode(prompt)
    if len(prompt_tokens) == 0:
        return "Please provide a valid prompt.", None

    start_tokens = prompt_tokens[np.newaxis, :]
    
    # Generate text
    generated_idx = transformer_model.generate(start_tokens, max_new_tokens=int(max_tokens), temperature=temperature)
    generated_text = tokenizer.decode(generated_idx[0])
    
    # After generation, we can do a forward pass of the full generated text
    # to capture attention weights of the last layer for visualization
    # We take up to max_seq_len tokens to avoid shape errors
    vis_tokens = generated_idx[0][-32:] 
    transformer_model.forward(vis_tokens[np.newaxis, :], training=False)
    
    # Extract attention weights from the last block
    # attn_weights shape: (batch, n_heads, seq, seq)
    # We'll average over heads for visualization
    attn = transformer_model.blocks[-1].attn._attn_weights[0] # (n_heads, seq, seq)
    attn_avg = np.mean(attn, axis=0) # (seq, seq)
    
    tokens_chars = [tokenizer.idx_to_char[t] for t in vis_tokens]
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(attn_avg, cmap="viridis", interpolation="nearest")
    
    # Only show ticks if sequence isn't too long, else it's unreadable
    if len(tokens_chars) <= 64:
        ax.set_xticks(np.arange(len(tokens_chars)))
        ax.set_yticks(np.arange(len(tokens_chars)))
        ax.set_xticklabels([repr(c).strip("'") for c in tokens_chars], rotation=90)
        ax.set_yticklabels([repr(c).strip("'") for c in tokens_chars])
    
    fig.colorbar(cax)
    plt.title("Attention Weights (Last Layer, Avg over Heads)")
    plt.tight_layout()
    
    return generated_text, fig

def get_results_df(filename):
    path = Path(__file__).parent / "results" / filename
    if path.exists():
        if filename.endswith(".csv"):
            return pd.read_csv(path)
        elif filename.endswith(".json"):
            with open(path, "r") as f:
                data = json.load(f)
            # convert dict of dicts to dataframe
            if isinstance(data, dict) and len(data) > 0 and isinstance(list(data.values())[0], dict):
                return pd.DataFrame.from_dict(data, orient='index').reset_index().rename(columns={'index': 'Variant'})
            return pd.DataFrame([data])
    return pd.DataFrame({"Message": ["No results found. Run benchmarks first."]})

with gr.Blocks(title="NeuraForge Demo", theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown("# 🚀 NeuraForge: Neural Network From Scratch in Pure NumPy")
    gr.Markdown("No PyTorch. No TensorFlow. No Autograd. Just math.")
    
    with gr.Tabs():
        with gr.TabItem("Transformer Demo"):
            gr.Markdown("### Character-level Transformer Text Generation (Tiny Shakespeare)")
            gr.Markdown("Watch a NumPy transformer generate Shakespeare and view its attention patterns.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_in = gr.Textbox(label="Prompt", value="HAMLET:\n", lines=3)
                    max_tokens = gr.Slider(minimum=10, maximum=300, value=100, step=10, label="Max New Tokens")
                    temp = gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature")
                    gen_btn = gr.Button("Generate Text", variant="primary")
                with gr.Column(scale=2):
                    out_text = gr.Textbox(label="Generated Text", lines=8)
            
            with gr.Row():
                attn_plot = gr.Plot(label="Attention Heatmap (Last Layer)")
                
            gen_btn.click(fn=generate_text, inputs=[prompt_in, max_tokens, temp], outputs=[out_text, attn_plot])
            
        with gr.TabItem("Digit Recognizer (MNIST)"):
            acc_text = ""
            acc_path = Path(__file__).parent / "results" / "demo_model_acc.json"
            if acc_path.exists():
                with open(acc_path, "r") as f:
                    data = json.load(f)
                    acc_text = f" **(Test Accuracy: {data['accuracy']*100:.2f}%)**"

            gr.Markdown(f"### Draw a digit (0-9) to see real-time inference using our custom NumPy framework.{acc_text}")
            gr.Markdown("Currently loading: `results/demo_model.npz`")
            with gr.Row():
                with gr.Column():
                    sketchpad = gr.Sketchpad(label="Draw here", type="numpy")
                    btn = gr.Button("Predict")
                with gr.Column():
                    label = gr.Label(num_top_classes=3)
                    
            btn.click(fn=predict_digit, inputs=sketchpad, outputs=label)
            
        with gr.TabItem("Benchmark Results"):
            gr.Markdown("### Phase 5: Scaling Study (NeuraForge vs PyTorch)")
            gr.Dataframe(get_results_df("scaling/scaling_results.json"))
            
            gr.Markdown("### Phase 6: Activation Ablation (ForageAct vs Baselines)")
            gr.Dataframe(get_results_df("ablations/activation_ablation.json"))
            
            gr.Markdown("### Phase 6: Optimizer Ablation (NeuroGrad vs Baselines)")
            gr.Dataframe(get_results_df("ablations/optimizer_ablation.json"))

if __name__ == "__main__":
    demo.launch()
