# Gradient Derivations

This document details the hand-derived gradients for key layers and operations. These mathematical truths form the basis of the operations in NeuraForge.

## 1. Softmax and Cross Entropy (Fused)

The softmax function for logits $z_i$ and the cross-entropy loss against one-hot targets $y_i$ for a single sample is:
$$ \hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}} $$
$$ L = -\sum_i y_i \log(\hat{y}_i) $$

Using the chain rule, the gradient of the loss with respect to a single logit $z_k$ is:
$$ \frac{\partial L}{\partial z_k} = \sum_i \frac{\partial L}{\partial \hat{y}_i} \frac{\partial \hat{y}_i}{\partial z_k} $$

For $i = k$: $\frac{\partial \hat{y}_k}{\partial z_k} = \hat{y}_k(1 - \hat{y}_k)$
For $i \neq k$: $\frac{\partial \hat{y}_i}{\partial z_k} = -\hat{y}_i \hat{y}_k$

Substituting into the sum:
$$ \frac{\partial L}{\partial z_k} = -y_k \frac{1}{\hat{y}_k} \hat{y}_k (1 - \hat{y}_k) - \sum_{i \neq k} y_i \frac{1}{\hat{y}_i} (-\hat{y}_i \hat{y}_k) $$
$$ \frac{\partial L}{\partial z_k} = -y_k + y_k \hat{y}_k + \sum_{i \neq k} y_i \hat{y}_k $$
$$ \frac{\partial L}{\partial z_k} = -y_k + \hat{y}_k \sum_i y_i = \hat{y}_k - y_k $$

Averaged over a batch of size $N$, the gradient is cleanly represented as:
$$ \frac{\partial L}{\partial Z} = \frac{1}{N} (\hat{Y} - Y) $$

## 2. Batch Normalization

For an input mini-batch $x$, with mean $\mu$ and variance $\sigma^2$:
$$ \mu = \frac{1}{m}\sum_{i=1}^m x_i $$
$$ \sigma^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu)^2 $$
$$ \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} $$
$$ y_i = \gamma \hat{x}_i + \beta $$

Given $\frac{\partial L}{\partial y_i}$, the gradients flow through three paths to $x_i$: directly through $\hat{x}_i$, and indirectly through $\mu$ and $\sigma^2$.

$$ \frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \gamma $$
$$ \frac{\partial L}{\partial \sigma^2} = \sum_{i=1}^m \frac{\partial L}{\partial \hat{x}_i} (x_i - \mu) \left(-\frac{1}{2}\right) (\sigma^2 + \epsilon)^{-3/2} $$
$$ \frac{\partial L}{\partial \mu} = \left(\sum_{i=1}^m \frac{\partial L}{\partial \hat{x}_i} \frac{-1}{\sqrt{\sigma^2 + \epsilon}}\right) + \frac{\partial L}{\partial \sigma^2} \frac{1}{m} \sum_{i=1}^m -2(x_i - \mu) $$

Combining these terms, the gradient with respect to the input $x_i$ is:
$$ \frac{\partial L}{\partial x_i} = \frac{1}{m \sqrt{\sigma^2 + \epsilon}} \left( m \frac{\partial L}{\partial \hat{x}_i} - \sum_{j=1}^m \frac{\partial L}{\partial \hat{x}_j} - \hat{x}_i \sum_{j=1}^m \frac{\partial L}{\partial \hat{x}_j} \hat{x}_j \right) $$

## 3. Layer Normalization

LayerNorm is mathematically similar to BatchNorm, but the mean and variance are computed across the feature dimension $D$ rather than the batch dimension $m$.

$$ \frac{\partial L}{\partial x_{i,d}} = \frac{1}{D \sqrt{\sigma_i^2 + \epsilon}} \left( D \frac{\partial L}{\partial \hat{x}_{i,d}} - \sum_{k=1}^D \frac{\partial L}{\partial \hat{x}_{i,k}} - \hat{x}_{i,d} \sum_{k=1}^D \frac{\partial L}{\partial \hat{x}_{i,k}} \hat{x}_{i,k} \right) $$

## 4. Scaled Dot-Product Attention

The forward pass is:
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

Let $S = \frac{QK^T}{\sqrt{d_k}}$ and $W = \text{softmax}(S)$. Then output $O = WV$.
Given $dO = \frac{\partial L}{\partial O}$:

1. **Gradient of V**:
$$ dV = W^T dO $$
2. **Gradient of W**:
$$ dW = dO V^T $$
3. **Gradient of S (through softmax)**:
Using the Jacobian of softmax, the gradient of the pre-softmax scores $S$ is:
$$ dS = W \odot (dW - \text{sum}(dW \odot W, \text{axis}=-1)) $$
4. **Gradients of Q and K**:
$$ dQ = \frac{dS K}{\sqrt{d_k}} $$
$$ dK = \frac{dS^T Q}{\sqrt{d_k}} $$
