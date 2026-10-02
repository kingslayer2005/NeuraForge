# NeuraForge Autograd Engine Architecture

NeuraForge uses a define-by-run, reverse-mode automatic differentiation engine.

## Core Design

The engine is built around the `Tensor` class, which wraps a NumPy array and tracks the operations that produced it.

1. **Graph Construction**: Every differentiable operation (e.g., `+`, `*`, `matmul`) creates a new `Tensor` and records the parent tensors and a `_backward_fn`. This dynamically builds a Directed Acyclic Graph (DAG) representing the computation.
2. **Reverse Topological Sort**: When `Tensor.backward()` is called, the engine performs a depth-first search to build a topologically sorted list of all tensors in the graph.
3. **Gradient Accumulation**: It then traverses this list in reverse, executing each tensor's `_backward_fn`. Gradients are accumulated using `+=` to handle diamond dependencies (where a single tensor is used in multiple downstream operations).
4. **Broadcasting Handling**: A critical helper function, `unbroadcast()`, ensures that gradients mapped back to broadcasted tensors are correctly summed along the broadcasted axes, maintaining shape correctness.

## Worked Example: `y = (X @ W) + b`

Consider a simplified Dense layer forward pass:

```python
X = Tensor(np.array([[1.0, 2.0]]), requires_grad=True) # shape (1, 2)
W = Tensor(np.array([[0.5], [-0.5]]), requires_grad=True) # shape (2, 1)
b = Tensor(np.array([0.1]), requires_grad=True) # shape (1,)

# Forward Pass
z1 = X @ W  # Tensor(_op="matmul", _parents=(X, W))
y = z1 + b  # Tensor(_op="add", _parents=(z1, b))
```

### Backward Pass

When we call `y.backward(np.ones_like(y.data))`:

1. **Topological Sort**: The order of nodes to visit in reverse is `[y, z1, X, W, b]`.
2. **Gradient for `y`**: initialized to `1.0`.
3. **`y._backward_fn()`**:
   - `y` is an addition `z1 + b`.
   - `z1.grad += unbroadcast(y.grad, z1.shape)` -> `z1.grad = 1.0`
   - `b.grad += unbroadcast(y.grad, b.shape)` -> `b.grad = 1.0`
4. **`z1._backward_fn()`**:
   - `z1` is a matmul `X @ W`.
   - `X.grad += z1.grad @ W.T` -> `1.0 @ [[0.5, -0.5]] = [[0.5, -0.5]]`
   - `W.grad += X.T @ z1.grad` -> `[[1.0], [2.0]] @ 1.0 = [[1.0], [2.0]]`
5. **Final Gradients**:
   - `X.grad` = `[[0.5, -0.5]]`
   - `W.grad` = `[[1.0], [2.0]]`
   - `b.grad` = `[1.0]`

## Memory Management

The graph is retained after `.backward()` unless explicitly broken or tensors fall out of scope. For inference, the `no_grad()` context manager disables graph construction entirely, saving memory and compute.
