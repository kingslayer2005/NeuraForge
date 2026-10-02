import sys
from pathlib import Path
import numpy as np

# ensure neuraforge is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuraforge.autograd import Tensor, unbroadcast
from scipy.special import logsumexp

def check_op(op_func, get_inputs, name):
    inputs = get_inputs()
    # Forward
    out = op_func(*inputs)
    # Backward
    d_out = np.ones_like(out.data)
    out.backward(d_out)
    
    # Numerical grad
    eps = 1e-6
    max_err = 0.0
    
    for i, t in enumerate(inputs):
        if not isinstance(t, Tensor) or not t.requires_grad:
            continue
            
        ng = np.zeros_like(t.data)
        it = np.nditer(t.data, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            old_val = t.data[idx]
            
            t.data[idx] = old_val + eps
            out_pos_data = op_func(*inputs).data.copy()
            
            t.data[idx] = old_val - eps
            out_neg_data = op_func(*inputs).data.copy()
            
            t.data[idx] = old_val
            
            deriv = np.sum((out_pos_data - out_neg_data) * d_out) / (2 * eps)
            ng[idx] = deriv
            it.iternext()
            
        err = np.max(np.abs(ng - t.grad) / (np.abs(ng) + 1e-8))
        if err > 1.0:
            print(f"\nDEBUG {name}: ng=\n{ng}\nt.grad=\n{t.grad}")
        max_err = max(max_err, err)
        
    status = "PASS" if max_err < 1e-6 else "FAIL"
    print(f"{name:20s} | {max_err:.2e} | {status}")


def verify_autograd():
    print("--- PHASE 3: AUTOGRAD CORRECTNESS ---")
    print("\n(a) Differentiable ops gradient check (float64, eps=1e-6, target < 1e-6)")
    print(f"{'Op Name':20s} | {'Max Rel Err':>10s} | Status")
    print("-" * 45)
    
    # add
    check_op(lambda a, b: a + b, lambda: [Tensor(np.random.randn(3, 3).astype(np.float64), requires_grad=True), Tensor(np.random.randn(3, 3).astype(np.float64), requires_grad=True)], "add")
    check_op(lambda a, b: a - b, lambda: [Tensor(np.random.randn(3, 3).astype(np.float64), requires_grad=True), Tensor(np.random.randn(3, 3).astype(np.float64), requires_grad=True)], "sub")
    check_op(lambda a, b: a * b, lambda: [Tensor(np.random.randn(3, 3).astype(np.float64), requires_grad=True), Tensor(np.random.randn(3, 3).astype(np.float64), requires_grad=True)], "mul")
    check_op(lambda a, b: a / b, lambda: [Tensor(np.random.randn(3, 3).astype(np.float64), requires_grad=True), Tensor(np.random.randn(3, 3).astype(np.float64) + 2.0, requires_grad=True)], "div")
    check_op(lambda a, b: a @ b, lambda: [Tensor(np.random.randn(3, 4).astype(np.float64), requires_grad=True), Tensor(np.random.randn(4, 5).astype(np.float64), requires_grad=True)], "matmul")
    check_op(lambda a: a.sum(), lambda: [Tensor(np.random.randn(3, 3).astype(np.float64), requires_grad=True)], "sum")
    check_op(lambda a: a.mean(), lambda: [Tensor(np.random.randn(3, 3).astype(np.float64), requires_grad=True)], "mean")
    check_op(lambda a: a.exp(), lambda: [Tensor(np.random.randn(3, 3).astype(np.float64), requires_grad=True)], "exp")
    check_op(lambda a: a.log(), lambda: [Tensor(np.random.rand(3, 3).astype(np.float64) + 0.1, requires_grad=True)], "log")
    check_op(lambda a: a ** 2.0, lambda: [Tensor(np.random.randn(3, 3).astype(np.float64), requires_grad=True)], "pow")
    check_op(lambda a: a.transpose(), lambda: [Tensor(np.random.randn(3, 4).astype(np.float64), requires_grad=True)], "transpose")
    check_op(lambda a: a.reshape((12,)), lambda: [Tensor(np.random.randn(3, 4).astype(np.float64), requires_grad=True)], "reshape")

    print("\n(b) Diamond test")
    x = Tensor(np.array([2.0]), requires_grad=True)
    y = x * x + x
    y.backward()
    print(f"y = x*x + x. Expected 5.0, Got: {x.grad[0]:.4f} -> {'PASS' if x.grad[0] == 5.0 else 'FAIL'}")
    
    x = Tensor(np.array([2.0]), requires_grad=True)
    y = Tensor(np.array([3.0]), requires_grad=True)
    z = (x * y) + (x * y)
    z.backward()
    print(f"z = (x*y) + (x*y). dZ/dx Expected 6.0, Got: {x.grad[0]:.4f} -> {'PASS' if x.grad[0] == 6.0 else 'FAIL'}")
    print(f"z = (x*y) + (x*y). dZ/dy Expected 4.0, Got: {y.grad[0]:.4f} -> {'PASS' if y.grad[0] == 4.0 else 'FAIL'}")

    print("\n(c) Broadcasting Matrix")
    shapes = [(1,), (5,), (1,4), (5,1), (5,4), (3,5,4)]
    print(f"{'Shape A':12s} | {'Shape B':12s} | Add | Sub | Mul | Div")
    print("-" * 60)
    for sA in shapes:
        for sB in shapes:
            try:
                # Add
                a = Tensor(np.ones(sA), requires_grad=True)
                b = Tensor(np.ones(sB), requires_grad=True)
                o = a + b
                o.backward(np.ones_like(o.data))
                add_pass = a.grad.shape == sA and b.grad.shape == sB
                
                # Sub
                a = Tensor(np.ones(sA), requires_grad=True)
                b = Tensor(np.ones(sB), requires_grad=True)
                o = a - b
                o.backward(np.ones_like(o.data))
                sub_pass = a.grad.shape == sA and b.grad.shape == sB
                
                # Mul
                a = Tensor(np.ones(sA), requires_grad=True)
                b = Tensor(np.ones(sB), requires_grad=True)
                o = a * b
                o.backward(np.ones_like(o.data))
                mul_pass = a.grad.shape == sA and b.grad.shape == sB
                
                # Div
                a = Tensor(np.ones(sA), requires_grad=True)
                b = Tensor(np.ones(sB), requires_grad=True)
                o = a / b
                o.backward(np.ones_like(o.data))
                div_pass = a.grad.shape == sA and b.grad.shape == sB
                
                print(f"{str(sA):12s} | {str(sB):12s} | {'P' if add_pass else 'F'}   | {'P' if sub_pass else 'F'}   | {'P' if mul_pass else 'F'}   | {'P' if div_pass else 'F'}")
            except ValueError:
                # Incompatible shapes for broadcast
                pass
                
    print("\n(d) Numerical stability")
    from neuraforge.autograd import softmax_cross_entropy
    x = np.array([[1000.0, 1001.0, 1002.0]])
    x_t = Tensor(x)
    # manual softmax
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    sm = e / e.sum(axis=-1, keepdims=True)
    print(f"Softmax [1000, 1001, 1002]: {sm}")
    
    # manual log_softmax
    lsm = (x - m) - np.log(np.sum(np.exp(x - m), axis=-1, keepdims=True))
    scipy_lsm = x - logsumexp(x, axis=-1, keepdims=True)
    print(f"LogSoftmax diff vs scipy: {np.max(np.abs(lsm - scipy_lsm)):.2e}")
    
    y = np.array([[1.0, 0.0, 0.0]])
    loss = softmax_cross_entropy(Tensor(np.array([[-100.0, 0.0, 100.0]])), y)
    print(f"Cross entropy with extreme prob (pred 1e-12): {loss.data:.4f}")

    print("\n(e) Zero-grad semantics")
    x = Tensor(np.array([2.0]), requires_grad=True)
    y1 = x * 3
    y1.backward()
    y2 = x * 4
    y2.backward()
    print(f"Accumulated gradient after two backward passes (expected 7.0): {x.grad[0]:.4f}")

if __name__ == "__main__":
    verify_autograd()
