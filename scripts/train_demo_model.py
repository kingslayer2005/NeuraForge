import argparse
from pathlib import Path
import json
import numpy as np

from neuraforge.data import train_val_test_split, Standardizer, DataLoader
from experiments.datasets import load_mnist
from neuraforge.model import Sequential
from neuraforge.layers import Dense
from neuraforge.activations import ReLU
from neuraforge.losses import SoftmaxCrossEntropy
from neuraforge.optimizers import Adam
from neuraforge.train import fit, accuracy_score
from neuraforge.io import save_model
from neuraforge.seed import seed_everything

def main():
    seed_everything(42)
    print("Fetching MNIST dataset...")
    X, y = load_mnist()
    Y = np.eye(10)[y]
    
    splits = train_val_test_split(X, Y, val_frac=0.1, test_frac=0.1, seed=42)
    scaler = Standardizer()
    X_train = scaler.fit_transform(splits["X_train"])
    X_val = scaler.transform(splits["X_val"])
    X_test = scaler.transform(splits["X_test"])
    y_train = splits["Y_train"]
    y_val = splits["Y_val"]
    y_test = splits["Y_test"]
    
    train_loader = DataLoader(X_train, y_train, batch_size=128, shuffle=True)
    val_loader = DataLoader(X_val, y_val, batch_size=128, shuffle=False)
    
    model = Sequential(
        Dense(784, 128),
        ReLU(),
        Dense(128, 128),
        ReLU(),
        Dense(128, 10)
    )
    
    loss_fn = SoftmaxCrossEntropy()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    print("Training demo model on MNIST...")
    history = fit(
        model, optimizer, loss_fn,
        train_loader, val_loader,
        epochs=3,
        verbose=True
    )
    
    # Evaluate
    preds = model.predict(X_test).argmax(axis=1)
    acc = accuracy_score(preds, y_test.argmax(axis=1))
    
    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / "demo_model.npz"
    
    save_model(model, str(out_path))
    print(f"Saved demo model to {out_path} with test accuracy: {acc*100:.2f}%")
    
    # Save the accuracy in a text file so app.py can read it
    acc_path = out_dir / "demo_model_acc.json"
    with open(acc_path, "w") as f:
        json.dump({"accuracy": acc}, f)

if __name__ == "__main__":
    main()
