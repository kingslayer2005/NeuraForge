"""
autograd.py — Reverse-mode automatic differentiation engine for NeuraForge.

This module provides a Tensor class that wraps NumPy arrays and automatically
tracks operations to build a computational graph. Calling .backward() on a
scalar Tensor walks the graph in reverse topological order, computing gradients
via the chain rule.

Design decisions:
    - The graph is built dynamically (define-by-run), like PyTorch.
    - After .backward(), the graph is NOT freed by default. Call .detach()
      explicitly to break the graph, or use no_grad() context manager.
    - Gradients accumulate with += (never =) to handle diamond dependencies.
    - Every binary op routes gradients through unbroadcast() to handle
      broadcasting-aware gradient accumulation correctly.
    - Gradients must be explicitly zeroed between training steps via zero_grad().

Implemented ops:
    Arithmetic: add, subtract, multiply, divide, matmul, power, neg
    Reductions: sum, mean, max
    Shape:      reshape, transpose, concatenate, getitem (slicing/indexing)
    Unary:      exp, log, sqrt, abs, tanh, sigmoid, relu, gelu, silu
    Composite:  softmax, log_softmax (numerically stable)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Helper: unbroadcast — the single most important function in this file
# ---------------------------------------------------------------------------
def unbroadcast(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Sum a gradient back to the shape of the tensor it came from.

    When NumPy broadcasts a (3,1) array against a (3,4) array, the result
    is (3,4). But the gradient for the (3,1) input must be summed over
    axis 1 (the broadcasted axis) to get back to shape (3,1).

    This function handles all broadcasting cases:
        - Extra leading dimensions added by broadcast (e.g. (4,) -> (3,4))
        - Dimensions of size 1 stretched (e.g. (3,1) -> (3,4))

    Parameters
    ----------
    grad : np.ndarray
        The gradient with the shape of the broadcasted result.
    target_shape : tuple
        The original shape of the tensor before broadcasting.

    Returns
    -------
    np.ndarray
        The gradient summed down to target_shape.
    """
    # If shapes already match, no work needed
    if grad.shape == target_shape:
        return grad

    # Step 1: Handle extra leading dimensions
    # If target has fewer dims, the leading axes were added by broadcasting.
    # Sum over those leading axes to remove them.
    ndim_diff = grad.ndim - len(target_shape)
    if ndim_diff > 0:
        # Sum over the extra leading axes (e.g. grad is (3,4,5) but target is (4,5))
        for _ in range(ndim_diff):
            grad = grad.sum(axis=0)

    # Step 2: Handle dimensions of size 1 that were stretched
    # For each axis where target_shape[i] == 1 but grad.shape[i] > 1,
    # we must sum over that axis and keep the dimension.
    for axis, (grad_size, target_size) in enumerate(zip(grad.shape, target_shape)):
        if target_size == 1 and grad_size > 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad


# ---------------------------------------------------------------------------
# No-grad context manager
# ---------------------------------------------------------------------------
class _NoGradContext:
    """Context manager to disable gradient tracking."""
    _enabled = False

    def __enter__(self):
        _NoGradContext._enabled = True
        return self

    def __exit__(self, *args):
        _NoGradContext._enabled = False


def no_grad():
    """Context manager that disables gradient tracking.

    Usage:
        with nf.no_grad():
            out = model(x)  # No graph is built
    """
    return _NoGradContext()


# ---------------------------------------------------------------------------
# Tensor — the core autodiff node
# ---------------------------------------------------------------------------
class Tensor:
    """A wrapper around a NumPy array that records operations for autodiff.

    Attributes
    ----------
    data : np.ndarray
        The actual numerical data.
    grad : np.ndarray or None
        The accumulated gradient after .backward(). None until backward is called.
    requires_grad : bool
        Whether this tensor participates in gradient computation.
    _backward_fn : callable or None
        A function that propagates gradients to this tensor's parents.
        Set by the operation that created this tensor.
    _parents : set of Tensor
        The tensors that this tensor was computed from.
        Used for topological sort during backward.
    _name : str
        Optional name for debugging.
    """

    def __init__(
        self,
        data: np.ndarray | float | list,
        requires_grad: bool = False,
        _parents: tuple = (),
        _op: str = "",
        name: str = "",
    ) -> None:
        # Convert to ndarray if not already
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data, dtype=np.float64)

        self.requires_grad = requires_grad

        # Gradient starts as None; accumulated during backward()
        self.grad: np.ndarray | None = None

        # The function that computes gradients for this node's parents
        self._backward_fn: Callable | None = None

        # Parent tensors in the computation graph
        self._parents: set[Tensor] = set(_parents)

        # Name of the op that produced this tensor (for debugging)
        self._op = _op

        # Optional human-readable name
        self._name = name

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def T(self) -> Tensor:
        """Shorthand for .transpose()."""
        return self.transpose()

    def __repr__(self) -> str:
        name_str = f", name='{self._name}'" if self._name else ""
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}{name_str})"

    def __len__(self) -> int:
        return len(self.data)

    # -------------------------------------------------------------------
    # Detach and clone
    # -------------------------------------------------------------------
    def detach(self) -> Tensor:
        """Return a new Tensor with the same data but no gradient history."""
        return Tensor(self.data.copy(), requires_grad=False)

    def clone(self) -> Tensor:
        """Return a copy of this tensor that shares no memory."""
        t = Tensor(self.data.copy(), requires_grad=self.requires_grad)
        return t

    def numpy(self) -> np.ndarray:
        """Return the underlying NumPy array (detached)."""
        return self.data

    def item(self) -> float:
        """Return the scalar value (only works for scalar tensors)."""
        return float(self.data.item())

    # -------------------------------------------------------------------
    # Zero grad
    # -------------------------------------------------------------------
    def zero_grad(self) -> None:
        """Reset the gradient to None."""
        self.grad = None

    # -------------------------------------------------------------------
    # Backward — reverse-mode autodiff
    # -------------------------------------------------------------------
    def backward(self, grad: np.ndarray | None = None) -> None:
        """Compute gradients via reverse-mode autodiff.

        Walks the computation graph in reverse topological order (DFS post-order)
        and calls each node's _backward_fn to propagate gradients.

        Parameters
        ----------
        grad : np.ndarray or None
            The gradient of the loss with respect to this tensor.
            If None (default for scalar tensors), uses ones_like(self.data).

        Notes
        -----
        - Gradients accumulate with +=, never =. This correctly handles
          diamond dependencies where one tensor feeds into multiple ops.
        - Topological order is computed via DFS post-order traversal with
          a visited set, NOT insertion order.
        """
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "backward() can only be called on a scalar tensor without "
                    f"providing grad. This tensor has shape {self.shape}."
                )
            # For scalar loss, gradient is 1.0
            grad = np.ones_like(self.data)

        # Initialize this tensor's gradient
        self.grad = grad

        # --- Build topological order via DFS post-order ---
        # We need to visit parents before children (in reverse order).
        # DFS post-order gives us children-last ordering; reversed gives
        # us the correct backward order (children first, parents last).
        topo_order: list[Tensor] = []
        visited: set[int] = set()

        def _build_topo(node: Tensor) -> None:
            """Depth-first post-order traversal."""
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)
            # Visit all parents first
            for parent in node._parents:
                _build_topo(parent)
            # Then add this node (post-order)
            topo_order.append(node)

        _build_topo(self)

        # --- Walk in reverse topological order ---
        # This ensures that when we process a node, all nodes that use it
        # as input have already propagated their gradients to it.
        for node in reversed(topo_order):
            if node._backward_fn is not None:
                node._backward_fn()

    # ===================================================================
    # ARITHMETIC OPERATIONS
    # ===================================================================

    def __add__(self, other: Tensor | float | np.ndarray) -> Tensor:
        """Element-wise addition: self + other.

        Backward:
            d(a+b)/da = 1, d(a+b)/db = 1
            Gradients pass through unchanged (after unbroadcasting).
        """
        other = _ensure_tensor(other)

        # Forward: element-wise add (NumPy handles broadcasting)
        out_data = self.data + other.data
        out = Tensor(out_data, requires_grad=(self.requires_grad or other.requires_grad),
                     _parents=(self, other), _op="+")

        if _NoGradContext._enabled:
            return out

        def _backward():
            if self.requires_grad:
                # Gradient of addition w.r.t. first operand is 1 * upstream_grad
                g = unbroadcast(out.grad, self.shape)
                # Accumulate (+=) to handle diamond dependencies
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                # Gradient of addition w.r.t. second operand is 1 * upstream_grad
                g = unbroadcast(out.grad, other.shape)
                other.grad = g if other.grad is None else other.grad + g

        out._backward_fn = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other: Tensor | float | np.ndarray) -> Tensor:
        """Element-wise subtraction: self - other.

        Backward:
            d(a-b)/da = 1, d(a-b)/db = -1
        """
        other = _ensure_tensor(other)

        out_data = self.data - other.data
        out = Tensor(out_data, requires_grad=(self.requires_grad or other.requires_grad),
                     _parents=(self, other), _op="-")

        if _NoGradContext._enabled:
            return out

        def _backward():
            if self.requires_grad:
                g = unbroadcast(out.grad, self.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                # Subtract: gradient w.r.t. second operand is -1 * upstream_grad
                g = unbroadcast(-out.grad, other.shape)
                other.grad = g if other.grad is None else other.grad + g

        out._backward_fn = _backward
        return out

    def __rsub__(self, other):
        other = _ensure_tensor(other)
        return other.__sub__(self)

    def __mul__(self, other: Tensor | float | np.ndarray) -> Tensor:
        """Element-wise multiplication: self * other.

        Backward:
            d(a*b)/da = b, d(a*b)/db = a
        """
        other = _ensure_tensor(other)

        out_data = self.data * other.data
        out = Tensor(out_data, requires_grad=(self.requires_grad or other.requires_grad),
                     _parents=(self, other), _op="*")

        if _NoGradContext._enabled:
            return out

        def _backward():
            if self.requires_grad:
                # d(a*b)/da = b * upstream_grad
                g = unbroadcast(out.grad * other.data, self.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                # d(a*b)/db = a * upstream_grad
                g = unbroadcast(out.grad * self.data, other.shape)
                other.grad = g if other.grad is None else other.grad + g

        out._backward_fn = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other: Tensor | float | np.ndarray) -> Tensor:
        """Element-wise division: self / other.

        Backward:
            d(a/b)/da = 1/b
            d(a/b)/db = -a/b^2
        """
        other = _ensure_tensor(other)

        out_data = self.data / other.data
        out = Tensor(out_data, requires_grad=(self.requires_grad or other.requires_grad),
                     _parents=(self, other), _op="/")

        if _NoGradContext._enabled:
            return out

        def _backward():
            if self.requires_grad:
                # d(a/b)/da = 1/b * upstream_grad
                g = unbroadcast(out.grad / other.data, self.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                # d(a/b)/db = -a/b^2 * upstream_grad
                g = unbroadcast(-out.grad * self.data / (other.data ** 2), other.shape)
                other.grad = g if other.grad is None else other.grad + g

        out._backward_fn = _backward
        return out

    def __rtruediv__(self, other):
        other = _ensure_tensor(other)
        return other.__truediv__(self)

    def __neg__(self) -> Tensor:
        """Negation: -self.

        Backward: d(-a)/da = -1
        """
        out_data = -self.data
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _parents=(self,), _op="neg")

        if _NoGradContext._enabled:
            return out

        def _backward():
            if self.requires_grad:
                g = -out.grad
                self.grad = g if self.grad is None else self.grad + g

        out._backward_fn = _backward
        return out

    def __pow__(self, exponent: float) -> Tensor:
        """Element-wise power: self ** exponent (scalar exponent only).

        Backward:
            d(a^n)/da = n * a^(n-1)
        """
        out_data = self.data ** exponent
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _parents=(self,), _op=f"**{exponent}")

        if _NoGradContext._enabled:
            return out

        def _backward():
            if self.requires_grad:
                # Power rule: d(x^n)/dx = n * x^(n-1)
                g = out.grad * exponent * (self.data ** (exponent - 1))
                self.grad = g if self.grad is None else self.grad + g

        out._backward_fn = _backward
        return out

    def __matmul__(self, other: Tensor) -> Tensor:
        """Matrix multiplication: self @ other.

        For 2D tensors A (N×K) and B (K×M):
            C = A @ B  →  C is (N×M)

        Backward:
            dL/dA = dL/dC @ B^T     (shape N×K)
            dL/dB = A^T @ dL/dC     (shape K×M)
        """
        other = _ensure_tensor(other)

        # Forward: standard matrix multiply
        out_data = self.data @ other.data  # shape: (N, M) for 2D inputs
        out = Tensor(out_data, requires_grad=(self.requires_grad or other.requires_grad),
                     _parents=(self, other), _op="@")

        if _NoGradContext._enabled:
            return out

        def _backward():
            if self.requires_grad:
                # dL/dA = dL/dC @ B^T
                if other.data.ndim == 1:
                    # When B is 1D, matmul treats it as a column vector
                    # out.grad shape: (N,), B shape: (K,)
                    # dL/dA = outer(dL/dC, B) → shape (N, K)
                    g = np.outer(out.grad, other.data)
                else:
                    g = out.grad @ other.data.T
                g = unbroadcast(g, self.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                # dL/dB = A^T @ dL/dC
                if self.data.ndim == 1:
                    g = np.outer(self.data, out.grad)
                else:
                    g = self.data.T @ out.grad
                g = unbroadcast(g, other.shape)
                other.grad = g if other.grad is None else other.grad + g

        out._backward_fn = _backward
        return out

    # ===================================================================
    # REDUCTION OPERATIONS
    # ===================================================================

    def sum(self, axis: int | tuple[int, ...] | None = None,
            keepdims: bool = False) -> Tensor:
        """Sum elements along axis.

        Backward:
            d(sum(x))/dx_i = 1 for all i
            The gradient is broadcast back to the original shape.
        """
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _parents=(self,), _op="sum")

        if _NoGradContext._enabled:
            return out

        # Cache the original shape for the backward pass
        original_shape = self.shape

        def _backward():
            if self.requires_grad:
                g = out.grad

                # If keepdims=False, we need to re-expand the gradient
                # to match the original shape before broadcasting.
                if not keepdims and axis is not None:
                    # Re-insert the summed axes as size-1 dimensions
                    if isinstance(axis, int):
                        axes = (axis,)
                    else:
                        axes = axis
                    for ax in sorted(axes):
                        g = np.expand_dims(g, axis=ax)

                # Broadcast the gradient to the original shape
                # (sum collapses dimensions; backward expands them back)
                g = np.broadcast_to(g, original_shape).copy()
                self.grad = g if self.grad is None else self.grad + g

        out._backward_fn = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None,
             keepdims: bool = False) -> Tensor:
        """Mean of elements along axis.

        Backward:
            d(mean(x))/dx_i = 1/N  where N is the number of elements being averaged.
        """
        out_data = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _parents=(self,), _op="mean")

        if _NoGradContext._enabled:
            return out

        original_shape = self.shape

        # Count the number of elements that were averaged
        if axis is None:
            n_elements = self.data.size
        else:
            if isinstance(axis, int):
                n_elements = self.data.shape[axis]
            else:
                n_elements = 1
                for ax in axis:
                    n_elements *= self.data.shape[ax]

        def _backward():
            if self.requires_grad:
                g = out.grad

                # Re-insert summed axes if they were collapsed
                if not keepdims and axis is not None:
                    if isinstance(axis, int):
                        axes = (axis,)
                    else:
                        axes = axis
                    for ax in sorted(axes):
                        g = np.expand_dims(g, axis=ax)

                # Scale by 1/N and broadcast to original shape
                g = np.broadcast_to(g / n_elements, original_shape).copy()
                self.grad = g if self.grad is None else self.grad + g

        out._backward_fn = _backward
        return out

    def max(self, axis: int | None = None,
            keepdims: bool = False) -> Tensor:
        """Maximum along axis.

        Backward:
            Gradient flows only to the element(s) that achieved the max value.
            This is a sparse routing — only the argmax position gets gradient.
        """
        out_data = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _parents=(self,), _op="max")

        if _NoGradContext._enabled:
            return out

        original_shape = self.shape

        def _backward():
            if self.requires_grad:
                g = out.grad

                # Re-expand the gradient if axes were collapsed
                if not keepdims and axis is not None:
                    g = np.expand_dims(g, axis=axis)
                    expanded_max = np.expand_dims(out.data, axis=axis)
                else:
                    expanded_max = out.data

                # Build a mask: 1 where input equals max, 0 elsewhere
                mask = (self.data == np.broadcast_to(expanded_max, original_shape))

                # If there are ties, distribute gradient equally
                # Count how many elements share the max along each slice
                count = mask.sum(axis=axis, keepdims=True)
                mask = mask / count  # Normalize by tie count

                # Route gradient only through max elements
                g = np.broadcast_to(g, original_shape) * mask
                self.grad = g if self.grad is None else self.grad + g

        out._backward_fn = _backward
        return out

    # ===================================================================
    # SHAPE OPERATIONS
    # ===================================================================

    def reshape(self, *shape) -> Tensor:
        """Reshape the tensor.

        Backward:
            Reshape the gradient back to the original shape.
        """
        # Handle both reshape(2,3) and reshape((2,3))
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        out_data = self.data.reshape(shape)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _parents=(self,), _op="reshape")

        if _NoGradContext._enabled:
            return out

        original_shape = self.shape

        def _backward():
            if self.requires_grad:
                # Simply reshape the gradient back
                g = out.grad.reshape(original_shape)
                self.grad = g if self.grad is None else self.grad + g

        out._backward_fn = _backward
        return out

    def transpose(self, *axes) -> Tensor:
        """Transpose (permute) the dimensions.

        Backward:
            Apply the inverse permutation to the gradient.
        """
        if len(axes) == 0:
            # Default: reverse all axes (like .T)
            perm = None
            out_data = self.data.T
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            perm = tuple(axes[0])
            out_data = np.transpose(self.data, perm)
        else:
            perm = axes
            out_data = np.transpose(self.data, perm)

        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _parents=(self,), _op="transpose")

        if _NoGradContext._enabled:
            return out

        def _backward():
            if self.requires_grad:
                if perm is None:
                    # Reverse of .T is .T
                    g = out.grad.T
                else:
                    # Inverse permutation: argsort gives the inverse
                    inv_perm = np.argsort(perm)
                    g = np.transpose(out.grad, inv_perm)
                self.grad = g if self.grad is None else self.grad + g

        out._backward_fn = _backward
        return out

    def __getitem__(self, index) -> Tensor:
        """Indexing/slicing: self[index].

        Backward:
            Place the gradient into a zero array at the indexed positions.
        """
        out_data = self.data[index]
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _parents=(self,), _op="getitem")

        if _NoGradContext._enabled:
            return out

        original_shape = self.shape

        def _backward():
            if self.requires_grad:
                # Create a zero gradient array matching the original shape
                g = np.zeros(original_shape, dtype=out.grad.dtype)
                # Place the upstream gradient at the indexed positions
                # np.add.at handles repeated indices correctly (accumulates)
                np.add.at(g, index, out.grad)
                self.grad = g if self.grad is None else self.grad + g

        out._backward_fn = _backward
        return out

    # ===================================================================
    # UNARY OPERATIONS
    # ===================================================================

    def exp(self) -> Tensor:
        """Element-wise exponential: exp(x).

        Backward: d(exp(x))/dx = exp(x)
        """
        out_data = np.exp(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _parents=(self,), _op="exp")

        if _NoGradContext._enabled:
            return out

        def _backward():
            if self.requires_grad:
                # d(exp(x))/dx = exp(x), which is just the output itself
                g = out.grad * out_data
                self.grad = g if self.grad is None else self.grad + g

        out._backward_fn = _backward
        return out

    def log(self) -> Tensor:
        """Element-wise natural logarithm: log(x).

        Backward: d(log(x))/dx = 1/x
        """
        out_data = np.log(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _parents=(self,), _op="log")

        if _NoGradContext._enabled:
            return out

        def _backward():
            if self.requires_grad:
                # d(log(x))/dx = 1/x
                g = out.grad / self.data
                self.grad = g if self.grad is None else self.grad + g

        out._backward_fn = _backward
        return out

    def sqrt(self) -> Tensor:
        """Element-wise square root: sqrt(x).

        Backward: d(sqrt(x))/dx = 1 / (2 * sqrt(x))
        """
        out_data = np.sqrt(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _parents=(self,), _op="sqrt")

        if _NoGradContext._enabled:
            return out

        def _backward():
            if self.requires_grad:
                # d(sqrt(x))/dx = 1/(2*sqrt(x)) = 0.5/sqrt(x)
                g = out.grad * 0.5 / out_data
                self.grad = g if self.grad is None else self.grad + g

        out._backward_fn = _backward
        return out

    def abs(self) -> Tensor:
        """Element-wise absolute value: |x|.

        Backward: d|x|/dx = sign(x)
        """
        out_data = np.abs(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _parents=(self,), _op="abs")

        if _NoGradContext._enabled:
            return out

        def _backward():
            if self.requires_grad:
                # sign(x) is -1, 0, or +1
                g = out.grad * np.sign(self.data)
                self.grad = g if self.grad is None else self.grad + g

        out._backward_fn = _backward
        return out

    def tanh(self) -> Tensor:
        """Element-wise tanh: tanh(x).

        Backward: d(tanh(x))/dx = 1 - tanh^2(x)
        """
        out_data = np.tanh(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _parents=(self,), _op="tanh")

        if _NoGradContext._enabled:
            return out

        def _backward():
            if self.requires_grad:
                # d(tanh(x))/dx = 1 - tanh^2(x), using cached output
                g = out.grad * (1.0 - out_data ** 2)
                self.grad = g if self.grad is None else self.grad + g

        out._backward_fn = _backward
        return out

    def sigmoid(self) -> Tensor:
        """Element-wise sigmoid: σ(x) = 1 / (1 + exp(-x)).

        Backward: d(σ(x))/dx = σ(x) * (1 - σ(x))
        """
        # Numerically stable sigmoid using the two-branch approach
        out_data = np.where(
            self.data >= 0,
            1.0 / (1.0 + np.exp(-self.data)),
            np.exp(self.data) / (1.0 + np.exp(self.data))
        )
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _parents=(self,), _op="sigmoid")

        if _NoGradContext._enabled:
            return out

        def _backward():
            if self.requires_grad:
                # σ'(x) = σ(x) * (1 - σ(x))
                g = out.grad * out_data * (1.0 - out_data)
                self.grad = g if self.grad is None else self.grad + g

        out._backward_fn = _backward
        return out

    def relu(self) -> Tensor:
        """Element-wise ReLU: max(0, x).

        Backward: d(relu(x))/dx = 1 if x > 0, else 0
        """
        out_data = np.maximum(0, self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _parents=(self,), _op="relu")

        if _NoGradContext._enabled:
            return out

        def _backward():
            if self.requires_grad:
                # Gradient flows only where input was positive
                mask = (self.data > 0).astype(self.data.dtype)
                g = out.grad * mask
                self.grad = g if self.grad is None else self.grad + g

        out._backward_fn = _backward
        return out

    def gelu(self) -> Tensor:
        """Element-wise GELU using the tanh approximation.

        f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

        Backward:
            f'(x) = 0.5 * (1 + tanh(g)) + 0.5 * x * sech^2(g) * g'(x)
            where g(x) = sqrt(2/π) * (x + 0.044715 * x^3)
              and g'(x) = sqrt(2/π) * (1 + 3 * 0.044715 * x^2)
        """
        SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)  # ≈ 0.7978845608
        COEFF = 0.044715

        x = self.data
        # Inner argument: g(x) = sqrt(2/π) * (x + 0.044715 * x^3)
        inner = SQRT_2_OVER_PI * (x + COEFF * x ** 3)
        # tanh of the inner argument
        tanh_inner = np.tanh(inner)
        # GELU output: 0.5 * x * (1 + tanh(g(x)))
        out_data = 0.5 * x * (1.0 + tanh_inner)

        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _parents=(self,), _op="gelu")

        if _NoGradContext._enabled:
            return out

        def _backward():
            if self.requires_grad:
                # g'(x) = sqrt(2/π) * (1 + 3 * 0.044715 * x^2)
                g_prime = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * x ** 2)
                # sech^2(g) = 1 - tanh^2(g)
                sech2 = 1.0 - tanh_inner ** 2
                # Full derivative: 0.5 * (1 + tanh(g)) + 0.5 * x * sech^2(g) * g'(x)
                grad_gelu = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * g_prime
                g = out.grad * grad_gelu
                self.grad = g if self.grad is None else self.grad + g

        out._backward_fn = _backward
        return out

    def silu(self) -> Tensor:
        """Element-wise SiLU (Swish): x * σ(x).

        Backward:
            d(x·σ(x))/dx = σ(x) + x·σ(x)·(1 - σ(x))
                         = σ(x)·(1 + x·(1 - σ(x)))
        """
        # Numerically stable sigmoid
        sig = np.where(
            self.data >= 0,
            1.0 / (1.0 + np.exp(-self.data)),
            np.exp(self.data) / (1.0 + np.exp(self.data))
        )
        out_data = self.data * sig

        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _parents=(self,), _op="silu")

        if _NoGradContext._enabled:
            return out

        def _backward():
            if self.requires_grad:
                # Product rule: d(x·σ(x))/dx = σ(x) + x·σ'(x) = σ(x) + x·σ(x)·(1-σ(x))
                grad_silu = sig + self.data * sig * (1.0 - sig)
                g = out.grad * grad_silu
                self.grad = g if self.grad is None else self.grad + g

        out._backward_fn = _backward
        return out

    # ===================================================================
    # COMPOSITE OPERATIONS (numerically stable)
    # ===================================================================

    def softmax(self, axis: int = -1) -> Tensor:
        """Numerically stable softmax along axis.

        softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

        Subtracts the row max before exponentiation to prevent overflow.

        Backward:
            The Jacobian of softmax is: diag(p) - p @ p^T
            For each sample: dL/dx_i = sum_j (dL/dy_j * y_j * (delta_ij - y_i))
                           = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
        """
        # Subtract max for numerical stability (does not change the result)
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        # Exponentiate
        exp_x = np.exp(shifted)
        # Normalize
        out_data = exp_x / np.sum(exp_x, axis=axis, keepdims=True)

        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _parents=(self,), _op="softmax")

        if _NoGradContext._enabled:
            return out

        def _backward():
            if self.requires_grad:
                # Efficient softmax backward without building the full Jacobian:
                # dL/dx_i = p_i * (dL/dp_i - sum_j(dL/dp_j * p_j))
                # where p = softmax output
                p = out_data  # softmax probabilities
                dp = out.grad  # upstream gradient dL/dp

                # sum_j(dL/dp_j * p_j) for each sample
                sum_dp_p = np.sum(dp * p, axis=axis, keepdims=True)

                # Final gradient: p * (dp - sum_dp_p)
                g = p * (dp - sum_dp_p)
                self.grad = g if self.grad is None else self.grad + g

        out._backward_fn = _backward
        return out

    def log_softmax(self, axis: int = -1) -> Tensor:
        """Numerically stable log-softmax via logsumexp.

        log_softmax(x) = x - logsumexp(x)
                       = x - max(x) - log(sum(exp(x - max(x))))

        This is more numerically stable than log(softmax(x)) because it
        avoids computing softmax (which can underflow) and then taking log
        (which amplifies the error).

        Backward:
            dL/dx_i = dL/d(log_softmax)_i - softmax(x)_i * sum_j(dL/d(log_softmax)_j)
        """
        # Logsumexp trick: max-subtract for stability
        x_max = np.max(self.data, axis=axis, keepdims=True)
        shifted = self.data - x_max
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
        out_data = shifted - log_sum_exp

        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _parents=(self,), _op="log_softmax")

        if _NoGradContext._enabled:
            return out

        # Precompute softmax for the backward pass
        softmax_probs = np.exp(out_data)

        def _backward():
            if self.requires_grad:
                # dL/dx_i = dL/d(log_p)_i - p_i * sum_j(dL/d(log_p)_j)
                # where p_i = softmax(x)_i
                sum_grad = np.sum(out.grad, axis=axis, keepdims=True)
                g = out.grad - softmax_probs * sum_grad
                self.grad = g if self.grad is None else self.grad + g

        out._backward_fn = _backward
        return out


# ===================================================================
# STANDALONE FUNCTIONS (convenience wrappers)
# ===================================================================

def concatenate(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    """Concatenate tensors along an axis.

    Backward:
        Split the gradient back into chunks matching original tensor sizes.
    """
    data_list = [t.data for t in tensors]
    out_data = np.concatenate(data_list, axis=axis)

    any_grad = any(t.requires_grad for t in tensors)
    out = Tensor(out_data, requires_grad=any_grad,
                 _parents=tuple(tensors), _op="concat")

    if _NoGradContext._enabled:
        return out

    # Record the sizes along the concatenation axis for splitting in backward
    sizes = [t.shape[axis] for t in tensors]

    def _backward():
        # Split the gradient along the concatenation axis
        splits = np.split(out.grad, np.cumsum(sizes[:-1]), axis=axis)
        for t, g_chunk in zip(tensors, splits):
            if t.requires_grad:
                t.grad = g_chunk if t.grad is None else t.grad + g_chunk

    out._backward_fn = _backward
    return out


def stack(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    """Stack tensors along a new axis.

    Backward:
        Unstack (split along the new axis and squeeze).
    """
    data_list = [t.data for t in tensors]
    out_data = np.stack(data_list, axis=axis)

    any_grad = any(t.requires_grad for t in tensors)
    out = Tensor(out_data, requires_grad=any_grad,
                 _parents=tuple(tensors), _op="stack")

    if _NoGradContext._enabled:
        return out

    n = len(tensors)

    def _backward():
        # Split along the stacked axis and squeeze
        splits = np.split(out.grad, n, axis=axis)
        for t, g_chunk in zip(tensors, splits):
            if t.requires_grad:
                g = np.squeeze(g_chunk, axis=axis)
                t.grad = g if t.grad is None else t.grad + g

    out._backward_fn = _backward
    return out


# ===================================================================
# LOSS FUNCTIONS (fused for numerical stability)
# ===================================================================

def softmax_cross_entropy(logits: Tensor, targets_onehot: np.ndarray) -> Tensor:
    """Fused softmax + cross-entropy loss for clean gradients.

    Combines softmax and cross-entropy into a single operation so that
    the backward pass produces the clean (p - y) / N gradient instead of
    routing through the full softmax Jacobian.

    Parameters
    ----------
    logits : Tensor, shape (N, C)
        Raw (unnormalized) class scores.
    targets_onehot : np.ndarray, shape (N, C)
        One-hot encoded ground-truth labels.

    Returns
    -------
    Tensor (scalar)
        Mean cross-entropy loss over the batch.
    """
    N = logits.shape[0]

    # Numerically stable softmax: subtract row max before exp
    shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    # Cross-entropy loss: -sum(y * log(p)) / N
    # Add epsilon to avoid log(0)
    loss_val = -np.sum(targets_onehot * np.log(probs + 1e-12)) / N

    out = Tensor(np.array(loss_val), requires_grad=logits.requires_grad,
                 _parents=(logits,), _op="softmax_ce")

    def _backward():
        if logits.requires_grad:
            # The beautiful fused gradient: (p - y) / N
            g = (probs - targets_onehot) / N
            logits.grad = g if logits.grad is None else logits.grad + g

    out._backward_fn = _backward
    return out


def mse_loss(predictions: Tensor, targets: np.ndarray) -> Tensor:
    """Mean squared error loss.

    L = mean((predictions - targets)^2)

    Backward: dL/d(pred) = 2 * (pred - targets) / total_elements
    """
    diff = predictions.data - targets
    loss_val = np.mean(diff ** 2)
    total = predictions.data.size

    out = Tensor(np.array(loss_val), requires_grad=predictions.requires_grad,
                 _parents=(predictions,), _op="mse")

    def _backward():
        if predictions.requires_grad:
            g = 2.0 * diff / total
            predictions.grad = g if predictions.grad is None else predictions.grad + g

    out._backward_fn = _backward
    return out


# ===================================================================
# UTILITY
# ===================================================================

def _ensure_tensor(x) -> Tensor:
    """Convert x to a Tensor if it isn't one already."""
    if isinstance(x, Tensor):
        return x
    return Tensor(x, requires_grad=False)


def zeros(*shape, requires_grad=False, dtype=np.float64) -> Tensor:
    """Create a tensor of zeros."""
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)


def ones(*shape, requires_grad=False, dtype=np.float64) -> Tensor:
    """Create a tensor of ones."""
    return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad)


def randn(*shape, requires_grad=False, dtype=np.float64) -> Tensor:
    """Create a tensor with random normal values."""
    return Tensor(np.random.randn(*shape).astype(dtype), requires_grad=requires_grad)


def from_numpy(arr: np.ndarray, requires_grad=False) -> Tensor:
    """Create a Tensor from a NumPy array (shares memory)."""
    return Tensor(arr, requires_grad=requires_grad)
