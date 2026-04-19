"""
NCU target: our attention_fwd_bf16 kernel, run-only pass.
Config matches FlashInfer baseline: B=8, H=12, N=1024, D=64.
"""
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from python.attention import attention, HAS_CUDA_EXT

assert HAS_CUDA_EXT, "CUDA extension not compiled; run: pip install -e ."

B, H, N, D = 8, 12, 1024, 64
device = "cuda"
dtype = torch.bfloat16

q = torch.randn(B, H, N, D, device=device, dtype=dtype)
k = torch.randn(B, H, N, D, device=device, dtype=dtype)
v = torch.randn(B, H, N, D, device=device, dtype=dtype)

# warmup
for _ in range(5):
    out = attention(q, k, v)
torch.cuda.synchronize()

# single run for NCU to capture
out = attention(q, k, v)
torch.cuda.synchronize()
print(f"done: {out.shape} {out.dtype}")
