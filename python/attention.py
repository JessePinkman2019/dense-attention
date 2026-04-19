"""
Dense Attention - Python 接口
"""

import torch
from torch.autograd import Function
import math

try:
    import attn_cuda
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False


# ============================================================
# PyTorch 参考实现（用于正确性验证）
# ============================================================

def attention_ref(q, k, v):
    """标准 PyTorch FP32 参考实现，用于验证 CUDA kernel 正确性。"""
    B, H, N, D = q.shape
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


# ============================================================
# Autograd Function
# ============================================================

class AttentionFunction(Function):

    @staticmethod
    def forward(ctx, q, k, v, scale):
        B, H, N, D = q.shape
        out = torch.empty_like(q)
        lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)

        if HAS_CUDA_EXT:
            attn_cuda.forward(q, k, v, out, lse, scale, True)
        else:
            out.copy_(attention_ref(q.float(), k.float(), v.float()).to(q.dtype))
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
            lse.copy_(torch.logsumexp(scores, dim=-1))

        ctx.save_for_backward(q, k, v, out, lse)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        scale = ctx.scale
        B, H, N, D = q.shape
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        if HAS_CUDA_EXT and hasattr(attn_cuda, 'backward'):
            attn_cuda.backward(dout, q, k, v, out, lse, dq, dk, dv, scale)
        else:
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
            attn = torch.softmax(scores, dim=-1)
            dv.copy_(torch.matmul(attn.transpose(-2, -1), dout.float()).to(dv.dtype))
            dattn = torch.matmul(dout.float(), v.float().transpose(-2, -1))
            ds = attn * (dattn - (dattn * attn).sum(dim=-1, keepdim=True)) * scale
            dq.copy_(torch.matmul(ds, k.float()).to(dq.dtype))
            dk.copy_(torch.matmul(ds.transpose(-2, -1), q.float()).to(dk.dtype))

        return dq, dk, dv, None


# ============================================================
# 主接口函数
# ============================================================

def attention(q, k, v):
    """
    Dense attention。

    参数：q, k, v: Tensor [B, H, N, D]，FP16/BF16/FP32
    返回：out: Tensor [B, H, N, D]
    """
    assert q.shape == k.shape == v.shape

    B, H, N, D = q.shape
    scale = 1.0 / math.sqrt(D)

    orig_dtype = q.dtype
    if orig_dtype == torch.float32:
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

    if HAS_CUDA_EXT and q.is_cuda:
        out = AttentionFunction.apply(q, k, v, scale)
    else:
        out = attention_ref(q.float(), k.float(), v.float()).to(q.dtype)

    return out.float() if orig_dtype == torch.float32 else out
