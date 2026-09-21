"""
NeuroForage - Version 1

A neural network implemented completely from scratch using NumPy.
This version includes:
- Fully connected (Dense) layers
- ReLU activation
- Softmax with cross-entropy loss
- Manual backpropagation
- Basic SGD training loop
- Accuracy tracking and loss visualization

This is the base engine that future NeuroForage features
(custom activations, optimizers, growth, pruning) will build on.
"""

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------
# Fully Connected Layer
# --------------------------------------
class Dense:
    def __init__(self, input_dim, output_dim):
        # He initialization for stable training with ReLU
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((1, output_dim))

    def forward(self, X):
        # Store input for use during backprop
        self.X = X
        return X @ self.W + self.b

    def backward(self, dZ):
        # Compute gradients with respect to weights and bias
        self.dW = self.X.T @ dZ
        self.db = np.sum(dZ, axis=0, keepdims=True)

        # Return gradient for previous layer
        return dZ @ self.W.T


# --------------------------------------
# ReLU Activation
# --------------------------------------
class ReLU:
    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dA):
        # Gradient passes only where Z > 0
        return dA * (self.Z > 0)


# --------------------------------------
# Softmax + Cross Entropy Loss
# --------------------------------------
class SoftmaxCrossEntropy:
    def forward(self, logits, Y):
        # Subtract max for numerical stability
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)
        self.Y = Y

        loss = -np.sum(Y * np.log(self.probs + 1e-9)) / Y.shape[0]
        return loss

    def backward(self):
        # Gradient simplifies nicely for softmax + cross entropy
        return (self.probs - self.Y) / self.Y.shape[0]


# --------------------------------------
# Multi-Layer Perceptron Model
# --------------------------------------
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.fc1 = Dense(input_dim, hidden_dim)
        self.act1 = ReLU()
        self.fc2 = Dense(hidden_dim, output_dim)
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, X, Y):
        Z1 = self.fc1.forward(X)
        A1 = self.act1.forward(Z1)
        Z2 = self.fc2.forward(A1)
        loss = self.loss_fn.forward(Z2, Y)
        return loss

    def backward(self):
        dZ2 = self.loss_fn.backward()
        dA1 = self.fc2.backward(dZ2)
        dZ1 = self.act1.backward(dA1)
        self.fc1.backward(dZ1)

    def predict(self, X):
        Z1 = self.fc1.forward(X)
        A1 = self.act1.forward(Z1)
        Z2 = self.fc2.forward(A1)
        return np.argmax(Z2, axis=1)


# --------------------------------------
# Accuracy Calculation
# --------------------------------------
def accuracy(model, X, Y):
    predictions = model.predict(X)
    true_labels = np.argmax(Y, axis=1)
    return np.mean(predictions == true_labels)


# --------------------------------------
# Training Loop (SGD)
# --------------------------------------
def train(model, X, Y, epochs=1000, lr=0.01):
    losses = []

    for epoch in range(epochs):
        loss = model.forward(X, Y)
        model.backward()

        # Basic SGD update
        model.fc1.W -= lr * model.fc1.dW
        model.fc1.b -= lr * model.fc1.db
        model.fc2.W -= lr * model.fc2.dW
        model.fc2.b -= lr * model.fc2.db

        losses.append(loss)

        if epoch % 100 == 0:
            acc = accuracy(model, X, Y)
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    return losses


# --------------------------------------
# Run Example
# --------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # Simple dummy dataset (for testing the engine)
    X = np.random.randn(200, 10)
    Y_raw = np.random.randint(0, 3, size=200)
    Y = np.eye(3)[Y_raw]

    model = MLP(input_dim=10, hidden_dim=32, output_dim=3)
    losses = train(model, X, Y, epochs=1000, lr=0.01)

    plt.plot(losses)
    plt.title("Training Loss - NeuroForage v1")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()