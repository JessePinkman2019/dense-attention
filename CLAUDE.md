# CLAUDE.md

> 目标：用 Claude 编写 CUDA kernel，性能超过 FlashInfer（dense attention，H800）。

## 环境

- GPU: NVIDIA H800 × 8，Driver 560.35.03，CUDA 12.6
- Python: `/opt/conda/bin/python`（torch 2.11.0+cu126，flashinfer 0.6.8）

## 运行

```bash
CUDA_VISIBLE_DEVICES=1 python tests/test_correctness.py
CUDA_VISIBLE_DEVICES=1 python tests/test_performance.py
CUDA_VISIBLE_DEVICES=1 python benchmarks/compare_flashinfer.py
```
