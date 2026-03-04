import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------
# Dense Layer
# --------------------------------------
class Dense:

    def __init__(self, input_dim, output_dim):

        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((1, output_dim))

    def forward(self, X):

        self.X = X

        return X @ self.W + self.b

    def backward(self, dZ):

        self.dW = self.X.T @ dZ
        self.db = np.sum(dZ, axis=0, keepdims=True)

        return dZ @ self.W.T


# --------------------------------------
# Trainable Activation
# --------------------------------------
class ForageAct:

    def __init__(self):

        self.alpha = 0.1

    def forward(self, Z):

        self.Z = Z

        self.sigmoid = 1 / (1 + np.exp(-Z))

        self.tanh = np.tanh(Z)

        return Z * self.sigmoid + self.alpha * self.tanh

    def backward(self, dA):

        d_sigmoid = self.sigmoid * (1 - self.sigmoid)

        term1 = self.sigmoid + self.Z * d_sigmoid

        d_tanh = 1 - self.tanh ** 2

        term2 = self.alpha * d_tanh

        dZ = dA * (term1 + term2)

        self.dalpha = np.sum(dA * self.tanh)

        return dZ


# --------------------------------------
# Softmax + Cross Entropy
# --------------------------------------
class SoftmaxCrossEntropy:

    def forward(self, logits, Y):

        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))

        self.probs = exp / np.sum(exp, axis=1, keepdims=True)

        self.Y = Y

        loss = -np.sum(Y * np.log(self.probs + 1e-9)) / Y.shape[0]

        return loss

    def backward(self):

        return (self.probs - self.Y) / self.Y.shape[0]


# --------------------------------------
# Optimizer with Gradient Clipping
# --------------------------------------
class NeuroGrad:

    def __init__(self, lr=0.01, beta=0.9, clip_value=1.0):

        self.lr = lr

        self.beta = beta

        self.clip_value = clip_value

        self.velocities = {}

    def clip_gradient(self, grad):

        norm = np.linalg.norm(grad)

        if norm > self.clip_value:

            grad = grad * (self.clip_value / norm)

        return grad

    def update(self, name, param, grad):

        grad = self.clip_gradient(grad)

        if name not in self.velocities:

            self.velocities[name] = np.zeros_like(param)

        v = self.velocities[name]

        v = self.beta * v + (1 - self.beta) * grad

        self.velocities[name] = v

        param = param - self.lr * v

        return param


# --------------------------------------
# MLP Model
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
# Accuracy
# --------------------------------------
def accuracy(model, X, Y):

    preds = model.predict(X)

    labels = np.argmax(Y, axis=1)

    return np.mean(preds == labels)


# --------------------------------------
# Training
# --------------------------------------
def train(model, X, Y, epochs=1000, lr=0.01):

    optimizer = NeuroGrad(lr=lr)

    losses = []

    for epoch in range(epochs):

        loss = model.forward(X, Y)

        model.backward()

        model.fc1.W = optimizer.update("fc1_W", model.fc1.W, model.fc1.dW)

        model.fc1.b = optimizer.update("fc1_b", model.fc1.b, model.fc1.db)

        model.fc2.W = optimizer.update("fc2_W", model.fc2.W, model.fc2.dW)

        model.fc2.b = optimizer.update("fc2_b", model.fc2.b, model.fc2.db)

        model.act1.alpha = optimizer.update("alpha", model.act1.alpha, model.act1.dalpha)

        losses.append(loss)

        if epoch % 100 == 0:

            acc = accuracy(model, X, Y)

            print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

            print("Alpha:", model.act1.alpha)

    return losses


# --------------------------------------
# Run Experiment
# --------------------------------------
if __name__ == "__main__":

    np.random.seed(42)

    X = np.random.randn(200, 10)

    Y_raw = np.random.randint(0, 3, size=200)

    Y = np.eye(3)[Y_raw]

    model = MLP(10, 32, 3)

    losses = train(model, X, Y)

    plt.plot(losses)

    plt.title("Training Loss - NeuroForage v4")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.show()