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
# Trainable Action
# --------------------------------------
class ForageAct:
    def __init__(self):
        self.alpha = 0.1  # trainable parameter

    def forward(self, Z):
        self.Z = Z
        self.sigmoid = 1 / (1 + np.exp(-Z))
        self.tanh = np.tanh(Z)
        return Z * self.sigmoid + self.alpha * self.tanh

    def backward(self, dA):
        # derivative of x * sigmoid(x)
        d_sigmoid = self.sigmoid * (1 - self.sigmoid)
        term1 = self.sigmoid + self.Z * d_sigmoid

        # derivative of alpha * tanh(x)
        d_tanh = 1 - self.tanh**2
        term2 = self.alpha * d_tanh

        dZ = dA * (term1 + term2)

        # gradient for alpha
        self.dalpha = np.sum(dA * self.tanh)

        return dZ
    


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
        self.act1 = ForageAct()
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
        model.act1.alpha -= lr * model.act1.dalpha

        losses.append(loss)

        if epoch % 100 == 0:
            acc = accuracy(model, X, Y)
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

        if epoch % 200 == 0:
            print("Alpha:", model.act1.alpha)

    return losses
    print("Alpha:", model.act1.alpha)


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