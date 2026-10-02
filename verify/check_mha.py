import torch
import torch.nn as tnn
import numpy as np

def check():
    mha = tnn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True, bias=False)
    print("in_proj_weight shape:", mha.in_proj_weight.shape) # Should be (48, 16)
    print("out_proj.weight shape:", mha.out_proj.weight.shape) # Should be (16, 16)
    
if __name__ == "__main__":
    check()
