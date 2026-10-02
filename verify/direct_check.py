import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from neuraforge.autograd import Tensor


def check_direct():
    print("Direct transpose check:")
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    y = x.transpose().sum()
    y.backward()
    if np.allclose(x.grad, np.ones((3, 4))):
        print("transpose PASS")
    else:
        print("transpose FAIL")
        print("Expected shape:", (3, 4), "Got:", x.grad.shape)
        
    print("Direct reshape check:")
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    y = x.reshape((12,)).sum()
    y.backward()
    if np.allclose(x.grad, np.ones((3, 4))):
        print("reshape PASS")
    else:
        print("reshape FAIL")
        print("Expected shape:", (3, 4), "Got:", x.grad.shape)

if __name__ == "__main__":
    check_direct()
