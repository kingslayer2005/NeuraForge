import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import predict_digit
from experiments.datasets import load_mnist


def main():
    X, y = load_mnist()
    # take a test image (say index 60000)
    img_array = X[-1].reshape(28, 28)
    label = y[-1]
    
    # img_array is [0, 255] with white digits on black background.
    # To simulate gradio sketchpad (black on white), we invert it:
    img_array = 255.0 - img_array
    
    probs = predict_digit(img_array)
    print(f"Actual label: {label}")
    print("Predicted probs:")
    for k, v in probs.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()
