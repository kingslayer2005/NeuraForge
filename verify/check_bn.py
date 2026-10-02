import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from neuraforge.nn import BatchNorm1d

def check_bn_eval():
    np.random.seed(42)
    torch.manual_seed(42)
    
    nf_bn = BatchNorm1d(10)
    pt_bn = torch.nn.BatchNorm1d(10)
    
    # Init running stats to zero and ones explicitly to ensure start match
    nf_bn.running_mean = np.zeros(10)
    nf_bn.running_var = np.ones(10)
    pt_bn.running_mean.data = torch.zeros(10)
    pt_bn.running_var.data = torch.ones(10)
    
    # Ensure momentum matches
    nf_bn.momentum = 0.1
    pt_bn.momentum = 0.1
    
    # Train pass 1
    x = np.random.randn(32, 10).astype(np.float32)
    x_pt = torch.tensor(x)
    
    nf_bn.forward(x, training=True)
    pt_bn(x_pt)
    
    print("PyTorch   mean:", pt_bn.running_mean.data.numpy()[:5])
    print("NeuraForge mean:", nf_bn.running_mean[:5])
    print("PyTorch   var:", pt_bn.running_var.data.numpy()[:5])
    print("NeuraForge var:", nf_bn.running_var[:5])

if __name__ == "__main__":
    check_bn_eval()
