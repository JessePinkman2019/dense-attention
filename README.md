# Sparse Mask Attention

针对短序列（<1K）+ 高稀疏度随机 mask（~75%）场景的 CUDA attention 算子。

## 项目结构

```
sparse_mask_attention/
├── csrc/                      # CUDA 核心实现
│   ├── sparse_attention.cu    # 稀疏注意力主 kernel
│   ├── sparse_attention.h     # 头文件
│   └── utils.cuh              # CUDA 工具函数
├── python/                    # Python 接口
│   ├── __init__.py
│   ├── sparse_attention.py    # 主接口
│   └── benchmark.py           # 性能测试工具
├── benchmarks/                # 基准测试脚本
│   ├── compare_flash.py       # vs Flash Attention
│   ├── compare_flashinfer.py  # vs FlashInfer
│   └── profile_mfu.py         # MFU 分析
├── tests/                     # 单元测试
│   ├── test_correctness.py    # 正确性验证
│   └── test_performance.py    # 性能测试
├── setup.py
├── requirements.txt
└── README.md
```

## 安装

```bash
pip install -r requirements.txt
pip install -e .
```

## 使用示例

```python
import torch
from sparse_mask_attention import sparse_attention

batch_size, num_heads, seq_len, head_dim = 8, 12, 512, 64
q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)

mask = torch.rand(batch_size, num_heads, seq_len, seq_len, device='cuda') > 0.75  # ~75% sparse

output = sparse_attention(q, k, v, mask)
```

## 运行测试

```bash
# 正确性验证
python tests/test_correctness.py

# 性能测试
python tests/test_performance.py

# 对比 Flash Attention
python benchmarks/compare_flash.py --seq_len 512 --batch_size 16

# 对比 FlashInfer
python benchmarks/compare_flashinfer.py --seq_len 512 --batch_size 16

# MFU 分析
python benchmarks/profile_mfu.py
```

## 基线性能对比

测试配置：B=64, H=12, N=1024, D=64，dense attention，GPU: H800

| 实现 | 精度 | 平均耗时 | 相对加速 |
|------|------|----------|----------|
| CPU 裸 PyTorch | FP32 | 1410.9 ms | 1x |
| GPU 裸 PyTorch | FP16 | 8.442 ms | **167x** |
| FlashAttention2 2.8.3 | FP16 | 0.704 ms | **2004x** |
| FlashInfer 0.6.8 BatchPrefill（目标） | FP16 | 0.534 ms | **2643x** |

> 三者均为 dense attention（无 mask），结果已验证数值一致。FlashInfer 是我们的优化目标基线。

## License

MIT License
