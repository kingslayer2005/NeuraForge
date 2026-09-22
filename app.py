import gradio as gr
import numpy as np
import pandas as pd
from pathlib import Path

# Fix path to import neuraforge
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuraforge.model import Sequential
from neuraforge.layers import Dense
from neuraforge.activations import ReLU
from neuraforge.io import load_model

# We will load a pre-trained model if available, or initialize a random one
model_path = Path(__file__).parent.parent / "results" / "demo_model.npz"
if model_path.exists():
    model = load_model(str(model_path))
else:
    # Dummy model for UI layout if weights are not yet generated
    model = Sequential(
        Dense(784, 128),
        ReLU(),
        Dense(128, 128),
        ReLU(),
        Dense(128, 10)
    )

def predict_digit(image):
    if image is None:
        return {str(i): 0.0 for i in range(10)}
        
    # Image is a dict with 'composite' key containing the drawn image in Gradio 4+ Sketchpad
    if isinstance(image, dict) and 'composite' in image:
        img_array = image['composite']
    else:
        img_array = image
        
    # Resize and preprocess to match MNIST (28x28, flattened, normalized)
    from PIL import Image
    # Convert RGBA to grayscale
    img_pil = Image.fromarray(img_array).convert('L')
    img_pil = img_pil.resize((28, 28))
    
    # Convert back to array
    img = np.array(img_pil)
    
    # Invert colors if necessary (drawing is usually black on white in some modes, 
    # but we want white on black for MNIST). 
    # Usually gradio sketchpad gives black background with white ink if configured.
    # We just normalize to 0-1
    img = img.astype(np.float32) / 255.0
    
    # Flatten
    X = img.reshape(1, 784).astype(np.float64)
    
    # Forward pass
    logits = model.forward(X)
    
    # Softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    probs = probs[0]
    
    # Format for Gradio Label component
    return {str(i): float(probs[i]) for i in range(10)}


def get_results_df(filename):
    path = Path(__file__).parent.parent / "results" / filename
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame({"Message": ["No results found. Run benchmarks first."]})


with gr.Blocks(title="NeuraForge Demo") as demo:
    gr.Markdown("# NeuraForge: Neural Network From Scratch in Pure NumPy")
    gr.Markdown("No PyTorch. No TensorFlow. No Autograd. Just math.")
    
    with gr.Tabs():
        with gr.TabItem("Interactive Demo"):
            acc_text = ""
            acc_path = Path(__file__).parent.parent / "results" / "demo_model_acc.json"
            if acc_path.exists():
                import json
                with open(acc_path, "r") as f:
                    data = json.load(f)
                    acc_text = f" **(Test Accuracy: {data['accuracy']*100:.2f}%)**"

            gr.Markdown(f"### Draw a digit (0-9) to see real-time inference using our custom NumPy framework.{acc_text}")
            gr.Markdown("Currently loading: `results/demo_model.npz`")
            with gr.Row():
                with gr.Column():
                    # Gradio 4 Sketchpad
                    sketchpad = gr.Sketchpad(label="Draw here", type="numpy")
                    btn = gr.Button("Predict")
                with gr.Column():
                    label = gr.Label(num_top_classes=3)
                    
            btn.click(fn=predict_digit, inputs=sketchpad, outputs=label)
            
        with gr.TabItem("Benchmark Results"):
            gr.Markdown("### Phase 3: Activation Ablation (MNIST)")
            gr.Dataframe(get_results_df("activation_ablation/MNIST/summary.csv"))
            
            gr.Markdown("### Phase 3: Optimizer Comparison (MNIST)")
            gr.Dataframe(get_results_df("optimizer_comparison/MNIST/summary.csv"))
            
            gr.Markdown("### Phase 5: Performance (PyTorch vs NeuraForge)")
            gr.Dataframe(get_results_df("performance/benchmark.csv"))

if __name__ == "__main__":
    demo.launch()
