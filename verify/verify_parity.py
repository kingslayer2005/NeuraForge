import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as tnn

sys.path.insert(0, str(Path(__file__).parent.parent))

from neuraforge.nn import Dense, Dropout, Conv2d, MaxPool2d, BatchNorm1d, BatchNorm2d, LayerNorm, Embedding
from neuraforge.transformer import scaled_dot_product_attention, MultiHeadAttention
from neuraforge.autograd import Tensor

def compare_layer(name, nf_layer, pt_layer, x_np, config):
    # Set weights identical
    if hasattr(nf_layer, 'W') and hasattr(pt_layer, 'weight'):
        pt_layer.weight.data = torch.tensor(nf_layer.W.data.T if "Dense" in name else nf_layer.W.data, dtype=torch.float32)
    if hasattr(nf_layer, 'b') and hasattr(pt_layer, 'bias') and pt_layer.bias is not None:
        pt_layer.bias.data = torch.tensor(nf_layer.b.data.reshape(-1), dtype=torch.float32)
        
    x_pt = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    
    # Forward
    training_mode = pt_layer.training if hasattr(pt_layer, 'training') else True
    out_nf = nf_layer.forward(x_np, training=training_mode)
    out_pt = pt_layer(x_pt)
    
    fwd_diff = np.max(np.abs(out_nf - out_pt.detach().numpy()))
    
    # Backward
    d_out = np.random.randn(*out_nf.shape).astype(np.float32)
    d_in_nf = nf_layer.backward(d_out)
    
    out_pt.backward(torch.tensor(d_out))
    d_in_pt = x_pt.grad.numpy()
    
    bwd_diff = np.max(np.abs(d_in_nf - d_in_pt))
    
    passed = "PASS" if fwd_diff < 1e-5 and bwd_diff < 1e-5 else "FAIL"
    print(f"{name:15s} | {config:25s} | {fwd_diff:.2e} | {bwd_diff:.2e} | {passed}")

def verify_parity():
    print("--- PHASE 4: PYTORCH PARITY ---")
    print(f"{'Layer':15s} | {'Config':25s} | {'Fwd Diff':>10s} | {'Bwd Diff':>10s} | Status")
    print("-" * 75)
    
    # Dense
    nf_dense = Dense(10, 20)
    pt_dense = tnn.Linear(10, 20)
    compare_layer("Dense", nf_dense, pt_dense, np.random.randn(32, 10).astype(np.float32), "10->20")
    
    # Dropout (eval mode)
    nf_drop = Dropout(0.5)
    pt_drop = tnn.Dropout(0.5)
    pt_drop.eval()
    x = np.random.randn(10, 10).astype(np.float32)
    out_nf = nf_drop.forward(x, training=False)
    out_pt = pt_drop(torch.tensor(x))
    fwd = np.max(np.abs(out_nf - out_pt.numpy()))
    print(f"{'Dropout':15s} | {'eval, p=0.5':25s} | {fwd:.2e} | {'N/A':>10s} | {'PASS' if fwd < 1e-5 else 'FAIL'}")
    
    # Embedding
    nf_emb = Embedding(100, 16)
    pt_emb = tnn.Embedding(100, 16)
    pt_emb.weight.data = torch.tensor(nf_emb.W.data, dtype=torch.float32)
    x = np.random.randint(0, 100, size=(32, 10))
    x_pt = torch.tensor(x, dtype=torch.long)
    out_nf = nf_emb.forward(x, training=True)
    out_pt = pt_emb(x_pt)
    fwd = np.max(np.abs(out_nf - out_pt.detach().numpy()))
    
    d_out = np.random.randn(*out_nf.shape).astype(np.float32)
    nf_emb.backward(d_out)
    out_pt.backward(torch.tensor(d_out))
    # We don't check input grad for embedding, we check weight grad
    w_grad_diff = np.max(np.abs(nf_emb.W.grad - pt_emb.weight.grad.numpy()))
    print(f"{'Embedding':15s} | {'100->16':25s} | {fwd:.2e} | {w_grad_diff:.2e} | {'PASS' if fwd < 1e-5 and w_grad_diff < 1e-5 else 'FAIL'}")
    
    # Conv2d
    configs = [(1, 0), (2, 0), (1, 1), (2, 1)] # (stride, padding)
    for s, p in configs:
        nf_conv = Conv2d(3, 8, kernel_size=3, stride=s, padding=p)
        pt_conv = tnn.Conv2d(3, 8, kernel_size=3, stride=s, padding=p)
        compare_layer("Conv2d", nf_conv, pt_conv, np.random.randn(4, 3, 16, 16).astype(np.float32), f"s={s}, p={p}")
        
    # MaxPool2d
    nf_pool = MaxPool2d(2, 2)
    pt_pool = tnn.MaxPool2d(2, 2)
    # PyTorch doesn't expose weights for pool
    x = np.random.randn(4, 3, 16, 16).astype(np.float32)
    x_pt = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    out_nf = nf_pool.forward(x, training=True)
    out_pt = pt_pool(x_pt)
    fwd = np.max(np.abs(out_nf - out_pt.detach().numpy()))
    d_out = np.random.randn(*out_nf.shape).astype(np.float32)
    d_in_nf = nf_pool.backward(d_out)
    out_pt.backward(torch.tensor(d_out))
    bwd = np.max(np.abs(d_in_nf - x_pt.grad.numpy()))
    print(f"{'MaxPool2d':15s} | {'2x2':25s} | {fwd:.2e} | {bwd:.2e} | {'PASS' if fwd < 1e-5 and bwd < 1e-5 else 'FAIL'}")
    
    # BatchNorm1d
    for mode in [True, False]:
        nf_bn = BatchNorm1d(10)
        pt_bn = tnn.BatchNorm1d(10)
        if not mode: pt_bn.eval()
        # Ensure running stats match
        nf_bn.running_mean = np.random.randn(10)
        nf_bn.running_var = np.abs(np.random.randn(10))
        pt_bn.running_mean.data = torch.tensor(nf_bn.running_mean, dtype=torch.float32)
        pt_bn.running_var.data = torch.tensor(nf_bn.running_var, dtype=torch.float32)
        compare_layer("BatchNorm1d", nf_bn, pt_bn, np.random.randn(32, 10).astype(np.float32), f"train={mode}")
        
    # BatchNorm2d
    for mode in [True, False]:
        nf_bn = BatchNorm2d(10)
        pt_bn = tnn.BatchNorm2d(10)
        if not mode: pt_bn.eval()
        nf_bn.running_mean = np.random.randn(10)
        nf_bn.running_var = np.abs(np.random.randn(10))
        pt_bn.running_mean.data = torch.tensor(nf_bn.running_mean, dtype=torch.float32)
        pt_bn.running_var.data = torch.tensor(nf_bn.running_var, dtype=torch.float32)
        compare_layer("BatchNorm2d", nf_bn, pt_bn, np.random.randn(4, 10, 8, 8).astype(np.float32), f"train={mode}")

    # LayerNorm
    nf_ln = LayerNorm(10)
    pt_ln = tnn.LayerNorm(10)
    compare_layer("LayerNorm", nf_ln, pt_ln, np.random.randn(32, 10).astype(np.float32), "")

    # MultiHeadAttention
    nf_mha = MultiHeadAttention(d_model=16, n_heads=4)
    pt_mha = tnn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
    
    # Map weights
    in_proj_weight = np.concatenate([nf_mha.W_q.data.T, nf_mha.W_k.data.T, nf_mha.W_v.data.T], axis=0)
    in_proj_bias = np.concatenate([nf_mha.b_q.data, nf_mha.b_k.data, nf_mha.b_v.data], axis=0)
    pt_mha.in_proj_weight.data = torch.tensor(in_proj_weight, dtype=torch.float32)
    pt_mha.in_proj_bias.data = torch.tensor(in_proj_bias, dtype=torch.float32)
    pt_mha.out_proj.weight.data = torch.tensor(nf_mha.W_o.data.T, dtype=torch.float32)
    pt_mha.out_proj.bias.data = torch.tensor(nf_mha.b_o.data, dtype=torch.float32)
    
    for use_mask in [False, True]:
        x_np = np.random.randn(2, 5, 16).astype(np.float32)
        x_pt = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
        
        mask_np = None
        mask_pt = None
        if use_mask:
            # Causal mask: upper triangular is True (mask out)
            mask_np = np.triu(np.ones((5, 5), dtype=bool), k=1)
            mask_pt = torch.tensor(mask_np)
            
        out_nf = nf_mha.forward(x_np, training=True, mask=mask_np)
        out_pt, _ = pt_mha(x_pt, x_pt, x_pt, need_weights=False, is_causal=use_mask, attn_mask=mask_pt if use_mask else None)
        
        fwd_diff = np.max(np.abs(out_nf - out_pt.detach().numpy()))
        
        d_out = np.random.randn(*x_np.shape).astype(np.float32)
        out_pt.backward(torch.tensor(d_out))
        d_in_nf = nf_mha.backward(d_out)
        
        bwd_diff = np.max(np.abs(d_in_nf - x_pt.grad.numpy()))
        
        status = "PASS" if max(fwd_diff, bwd_diff) < 1e-5 else "FAIL"
        print(f"{'MultiHeadAttn':15s} | {'mask=' + str(use_mask):25s} | {fwd_diff:>10.2e} | {bwd_diff:>10.2e} | {status}")

if __name__ == "__main__":
    verify_parity()
