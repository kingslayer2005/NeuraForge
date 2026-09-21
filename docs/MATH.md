# NeuraForge Mathematical Foundations

This document derives the exact mathematics implemented in NeuraForge, matching the Python code line for line.

## 1. Layers

### 1.1 Dense (Fully Connected) Layer
Given an input $X \in \mathbb{R}^{N \times D_{in}}$, weight matrix $W \in \mathbb{R}^{D_{in} \times D_{out}}$, and bias $b \in \mathbb{R}^{1 \times D_{out}}$:

**Forward Pass:**
$$ Z = X W + b $$

**Backward Pass:**
Let $\frac{\partial L}{\partial Z} = dZ \in \mathbb{R}^{N \times D_{out}}$ be the gradient of the loss with respect to the output.

$$ \frac{\partial L}{\partial X} = dZ W^T $$
$$ \frac{\partial L}{\partial W} = X^T dZ $$
$$ \frac{\partial L}{\partial b} = \sum_{i=1}^N dZ_i $$

*(In code: `dW = X.T @ d_out`, `db = np.sum(d_out, axis=0, keepdims=True)`, `d_input = d_out @ self.W.data.T`)*

---

## 2. Activations

### 2.1 Standard Activations

#### ReLU
**Forward:** $f(z) = \max(0, z)$
**Backward:** $f'(z) = 1$ if $z > 0$ else $0$

#### Sigmoid ($\sigma$)
**Forward:** $\sigma(z) = \frac{1}{1 + e^{-z}}$
**Backward:** $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

#### Tanh
**Forward:** $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
**Backward:** $\tanh'(z) = 1 - \tanh^2(z)$

#### SiLU / Swish
**Forward:** $f(z) = z \cdot \sigma(z)$
**Backward:** $f'(z) = f(z) + \sigma(z)(1 - f(z))$

---

### 2.2 ForageAct
NeuraForge introduces `ForageAct`, an activation function with a learnable parameter $\alpha$:
$$ f(z) = z \cdot \sigma(z) + \alpha \cdot \tanh(z) $$

Where $\alpha$ can be a fixed scalar, a learnable scalar, or a learnable vector (one per neuron).

**Forward Pass:**
$$ \text{out} = z \cdot \sigma(z) + \alpha \cdot \tanh(z) $$

**Backward Pass (with respect to input $z$):**
Using the product rule and derivatives of Sigmoid and Tanh from above:
$$ \frac{\partial f}{\partial z} = \left[ z \cdot \sigma(z) \cdot (1 - \sigma(z)) + \sigma(z) \right] + \alpha \cdot [1 - \tanh^2(z)] $$

Given upstream gradient $dZ$:
$$ dZ_{in} = dZ \odot \frac{\partial f}{\partial z} $$

**Backward Pass (with respect to $\alpha$):**
$$ \frac{\partial f}{\partial \alpha} = \tanh(z) $$
$$ \frac{\partial L}{\partial \alpha} = \sum_{N} \left( dZ \odot \tanh(z) \right) $$

*(If $\alpha$ is a scalar, we sum over both batch and feature dimensions. If $\alpha$ is per-neuron, we sum only over the batch dimension).*

---

## 3. Losses

### 3.1 Softmax Cross-Entropy
Given raw logits $Z \in \mathbb{R}^{N \times C}$ and one-hot targets $Y \in \mathbb{R}^{N \times C}$:

**Forward Pass:**
1. Stability shift: $\hat{Z} = Z - \max(Z, \text{axis}=1)$
2. Softmax: $P = \frac{e^{\hat{Z}}}{\sum e^{\hat{Z}}}$
3. Loss: $L = -\frac{1}{N} \sum_{i=1}^N \sum_{j=1}^C Y_{i,j} \log(P_{i,j})$

**Backward Pass:**
The beautiful simplification of Softmax + Cross-Entropy:
$$ \frac{\partial L}{\partial Z} = \frac{1}{N} (P - Y) $$

---

## 4. Optimizers

### 4.1 NeuroGrad
NeuroGrad uses Exponential Moving Average (EMA) momentum and gradient-norm clipping.

Let $g_t$ be the gradient at step $t$.

**1. Gradient Clipping**
If `clip_mode` is "per_tensor":
$$ g_t = g_t \cdot \min\left(1, \frac{\text{clip\_value}}{\|g_t\|_2}\right) $$

**2. EMA Momentum**
Unlike classical momentum ($v_t = \beta v_{t-1} + g_t$), EMA momentum takes a convex combination:
$$ v_t = \beta v_{t-1} + (1 - \beta) g_t $$

**3. Parameter Update**
$$ \theta_t = \theta_{t-1} - \eta \cdot v_t $$

**Equivalence Note**:
Without clipping, notice that:
$$ \theta_t = \theta_{t-1} - \eta (\beta v_{t-1} + (1-\beta)g_t) $$
Let $\tilde{v}_t$ be the classical momentum vector. Then $v_t = (1-\beta)\tilde{v}_t$.
$$ \theta_t = \theta_{t-1} - \eta (1-\beta) \tilde{v}_t $$
This demonstrates that EMA momentum is mathematically identical to classical momentum operating at an effective learning rate of $\eta_{eff} = \eta (1-\beta)$.
