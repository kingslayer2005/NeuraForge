import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from neuraforge.data import train_val_test_split, Standardizer, DataLoader
from experiments.datasets import load_mnist
from neuraforge.model import Sequential
from neuraforge.nn import Conv2d, MaxPool2d
from neuraforge.layers import Dense
from neuraforge.activations import ReLU
from neuraforge.losses import SoftmaxCrossEntropy
from neuraforge.optimizers import Adam
from neuraforge.train import fit, accuracy_score
from neuraforge.io import save_model
from neuraforge.seed import seed_everything

# Reshape layer wrapper
class Reshape:
    def __init__(self, *shape):
        self.shape = shape
    def forward(self, X, training=False):
        return X.reshape(X.shape[0], *self.shape)
    def backward(self, d_out):
        return d_out.reshape(-1, *self.in_shape[1:])
    def parameters(self):
        return []
    @property
    def config(self):
        return {"type": "Reshape", "shape": self.shape}
    # Wait, the Sequential module doesn't handle Reshape natively if not a Module
    # NeuraForge might just need a proper reshape layer, let's build it properly

from neuraforge.nn import Module
class Flatten(Module):
    def forward(self, X, training=False):
        self._in_shape = X.shape
        return X.reshape(X.shape[0], -1)
    def backward(self, d_out):
        return d_out.reshape(self._in_shape)
    def parameters(self):
        return []
    @property
    def config(self):
        return {"type": "Flatten"}

class ReshapeInput(Module):
    def forward(self, X, training=False):
        self._in_shape = X.shape
        # Input to CNN should be (N, C, H, W)
        return X.reshape(X.shape[0], 1, 28, 28)
    def backward(self, d_out):
        return d_out.reshape(self._in_shape)
    def parameters(self):
        return []
    @property
    def config(self):
        return {"type": "ReshapeInput"}

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
    
    # A simple CNN: 1 channel -> 8 -> pool -> 16 -> pool -> flatten -> dense
    model = Sequential(
        ReshapeInput(),
        Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2d(kernel_size=2, stride=2),
        Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2d(kernel_size=2, stride=2),
        Flatten(),
        Dense(16 * 7 * 7, 128),
        ReLU(),
        Dense(128, 10)
    )
    
    loss_fn = SoftmaxCrossEntropy()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    from neuraforge.train import EarlyStopping
    early_stop = EarlyStopping(patience=3)
    
    print("Training real CNN on MNIST...")
    history = fit(
        model, optimizer, loss_fn,
        train_loader, val_loader,
        epochs=15, 
        early_stopping=early_stop,
        verbose=True
    )
    
    print("Full per-epoch curve:")
    for ep in range(len(history['val_acc'])):
        print(f"Epoch {ep:2d} | Train Acc: {history['train_acc'][ep]*100:.2f}% | Val Acc: {history['val_acc'][ep]*100:.2f}%")
        
    # Evaluate
    preds = model.predict(X_test).argmax(axis=1)
    acc = accuracy_score(preds, y_test.argmax(axis=1))
    print(f"Final CNN Test Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
