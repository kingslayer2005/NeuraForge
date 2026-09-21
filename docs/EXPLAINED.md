# NeuraForge Code Explained

This document explains the architecture of the NeuraForge codebase, serving as a study guide for interviews.

## Core Philosophy

NeuraForge strictly adheres to an object-oriented approach for building neural networks, mirroring the structural design of major frameworks like PyTorch, but with every primitive implemented explicitly in NumPy.

1. **Parameters are Objects**: Instead of raw dictionaries, weights and biases are wrapped in a `Parameter` class holding `.data` and `.grad`.
2. **Stateful Forward, Stateless Backward**: Layers remember their inputs during the `forward` pass so they can compute gradients during the `backward` pass.
3. **Registry Pattern**: The `Sequential` model recursively collects all `Parameter` objects from its layers, assigning them unique names (e.g. `layer_0/W`) so optimizers can track their state.

## Module Breakdown

### 1. `model.py` (The Container)
The `Sequential` class holds a list of layers. 
- `forward(X)`: Loops through layers, passing the output of layer $i$ as the input to layer $i+1$.
- `backward(d_out)`: Loops through layers in *reverse*, passing the gradient from layer $i$ as the upstream gradient to layer $i-1$.
- `parameters()`: Returns a flat list of all learnable `Parameter` objects in the network.

### 2. `layers.py` (The Transformations)
The `Dense` layer performs the core affine transformation $Z = XW + b$.
- **Initialization**: Uses He initialization ($\mathcal{N}(0, \sqrt{2/fan\_in})$) which is optimal for ReLU-family activations to prevent vanishing/exploding gradients.
- **Forward**: Caches the input `X` as `self.input`.
- **Backward**: Uses `self.input` and the upstream gradient `d_out` to compute `self.W.grad` and `self.b.grad`, and returns the gradient with respect to the input to pass down the network.

### 3. `activations.py` (The Non-Linearities)
Standard activations (ReLU, Sigmoid, etc.) are implemented without learnable parameters.

**ForageAct** is our custom activation: $f(z) = z \cdot \sigma(z) + \alpha \cdot \tanh(z)$.
- It inherits from the base `Layer` class.
- In `mode="scalar"`, $\alpha$ is a `Parameter` of shape `(1,)`.
- In `mode="per_neuron"`, $\alpha$ is a `Parameter` of shape `(N,)`.
- The `backward` pass computes the gradient with respect to $\alpha$ (which is $\sum \tanh(z) \cdot dZ$) and stores it in `alpha.grad`, while returning the gradient with respect to $z$.

### 4. `losses.py` (The Objective)
`SoftmaxCrossEntropy` combines the softmax activation and the cross-entropy loss into a single class for numerical stability.
- **Why?** Computing softmax probabilities can easily overflow floating-point limits. We subtract the maximum logit before exponentiation: $e^{z - \max(z)}$.
- **Gradients**: The combined gradient is beautifully simple: $\frac{1}{N} (\hat{y} - y)$, where $\hat{y}$ are the predicted probabilities and $y$ are the one-hot targets.

### 5. `optimizers.py` (The Updates)
Optimizers hold a reference to the list of `Parameter` objects returned by `model.parameters()`.
- `zero_grad()`: Sets `.grad` to zero for all parameters.
- `step()`: Updates `.data` based on `.grad`.
- **MomentumSGD**: Maintains a dictionary `self.velocities` keyed by `id(param)` to store the running velocity for each parameter.
- **NeuroGrad**: Implements Exponential Moving Average (EMA) momentum and global/per-tensor gradient clipping.

### 6. `train.py` (The Loop)
The `fit` function coordinates the entire process:
1. Iterate over epochs.
2. Iterate over mini-batches from `DataLoader`.
3. Zero gradients $\rightarrow$ Forward $\rightarrow$ Loss $\rightarrow$ Backward $\rightarrow$ Step.
4. Evaluate on validation set.
5. Check `EarlyStopping`. If validation loss stops improving for `patience` epochs, stop training and load the weights from the best epoch.

### 7. `growth.py` & `pruning.py` (The Evolution)
- **Net2WiderNet**: Widens a layer by copying existing neurons. It scales the outgoing weights of the next layer by $1/\text{count}$ to ensure the network output remains mathematically identical.
- **Pruning**: Computes neuron importance using a first-order Taylor approximation (outgoing weight magnitude $\times$ gradient). It removes the lowest-scoring columns of the current layer's weight matrix and the corresponding rows of the next layer's weight matrix.
