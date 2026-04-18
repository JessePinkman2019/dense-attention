"""
性能测量工具
"""

import torch
import math


def get_h800_peak_tflops(dtype=torch.bfloat16):
    if dtype in (torch.float16, torch.bfloat16):
        return 989.0  # H800 BF16/FP16 Tensor Core peak
    elif dtype == torch.float32:
        return 67.0   # H800 FP32
    else:
        return 989.0


def attention_flops(B, H, N, D, is_training=False):
    """注意力的理论 FLOPs（dense，不考虑稀疏）"""
    fwd_flops = 4 * B * H * N * N * D  # QK^T + AV
    if is_training:
        return fwd_flops * 3.5
    return fwd_flops


def measure_mfu(fn, B, H, N, D, is_training=False,
                warmup=10, repeat=50, dtype=torch.bfloat16):
    """
    测量函数的延迟和 MFU（基于 dense attention FLOPs）。

    返回：
        dict: {latency_ms, tflops, mfu_percent}
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / repeat
    elapsed_s = elapsed_ms / 1000.0

    flops = attention_flops(B, H, N, D, is_training)
    tflops = flops / elapsed_s / 1e12

    peak = get_h800_peak_tflops(dtype)
    mfu = tflops / peak * 100.0

    return {
        "latency_ms": elapsed_ms,
        "tflops": tflops,
        "mfu_percent": mfu,
        "peak_tflops": peak,
    }
