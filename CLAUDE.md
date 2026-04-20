# Dense Attention CUDA Kernel 优化

> **目标**：H800 BF16 dense attention latency < 0.311ms @ B=8,N=1024（超过 FlashInfer 0.6.8）
> **完整准则**：严格按照 `/root/.claude/plans/attention-cuda-flashinfer-plan-chj-home-compiled-bachman.md` 执行

## 环境

- GPU: H800，CUDA 12.6，固定用 `CUDA_VISIBLE_DEVICES=1`
- Python: `/opt/conda/bin/python`（torch 2.11.0+cu126，flashinfer 0.6.8）

## 启动协议（每个新 session 必须先执行）

```bash
python3 -c "import json,os; events=json.load(open('optimization_session.json')) if os.path.exists('optimization_session.json') else []; [print(json.dumps(e,indent=2,ensure_ascii=False)) for e in events[-3:]] or print('第 0 轮，尚未开始')"
cat csrc/attention.cu
cat ROUND_PLAN.json 2>/dev/null || echo "ROUND_PLAN.json 不存在"
```

## 角色（各自独立 Agent，上下文不共享）

三个角色均以**异步子 Agent**（`Agent` tool，`run_in_background=false`）方式启动，每个 Agent 只读持久化文件（session log / ROUND_PLAN.json / attention.cu），不依赖对话历史：

| 角色 | 触发方式 | subagent_type | 输入 | 输出 |
|------|----------|---------------|------|------|
| `/planner` | `Agent` tool | `general-purpose` | `optimization_session.json`（最近3轮） | `ROUND_PLAN.json` |
| `/generator` | `Agent` tool | `general-purpose` | `ROUND_PLAN.json` + `csrc/attention.cu` | 新 `attention.cu` + git commit |
| `/evaluator` | `Agent` tool | `general-purpose` | 最新 git commit | `optimization_session.json` 追加一轮 + Memory 更新 |

**顺序**：planner → generator → evaluator → planner → …（严格串行，不并行）

**强制要求**：
- 每个角色**必须**通过 `Agent` tool 以独立子 Agent 启动，禁止在主对话中直接执行角色逻辑。
- 子 Agent prompt 必须包含：角色名、输入文件路径、输出文件路径、具体执行指令。
- 主 Agent 只负责串行调度：等待上一个子 Agent 完成后再启动下一个。

> **上下文节省关键**：每次只开一个角色 Agent，完成后关闭。下一角色读持久化文件，不看聊天记录。

### ⚠️ 角色分离失效的教训（Round 12-13 反面案例）

**问题**：Round 12-13 中主 Agent 在同一对话里同时扮演三个角色，没有通过 `Agent` tool 启动独立子 Agent。导致：

1. **evaluator 没有独立视角**：自己评估自己的 kernel，缺乏外部约束，导致在 WGMMA 错误路径上调试 10+ 小时而没有及早止损。
2. **NCU 分析缺失**：正确的 evaluator 应在第一轮就运行 NCU，发现 B128 WGMMA 内存访问模式异常，而不是靠大量试错。
3. **上下文膨胀**：单一 Agent 积累了大量中间调试状态，判断力下降，无法从全局视角识别"这条路根本走不通"。
4. **"自己评估自己"偏差**：generator 实现的方案，由同一个 Agent 来评估，天然倾向于"再试一次"而非"放弃这条路"。

**Evaluator 强制 Checklist（每轮必须全部完成，不得跳过任何步骤）**：

**Step 1 — 编译**
```bash
CUDA_VISIBLE_DEVICES=1 pip install -e . -q 2>&1 | tail -3
```
- 编译失败 → 立即停止，在 session log 中记录 `compile_success: false`，通知 planner 修复。

**Step 2 — 正确性（必须用真实随机 Q/K/V，不得用 V=1 等退化情况）**
```bash
CUDA_VISIBLE_DEVICES=1 python tests/test_correctness.py
```
- 任何 FAIL → 立即停止，记录 `correctness_pass: false`，禁止进行性能测试。
- `vs_flashinfer_ratio > 1`（声称超过 FlashInfer）→ 必须标记为"待验证异常"，单独确认。

**Step 3 — 延迟（B=8 config 对标基线）**
```bash
CUDA_VISIBLE_DEVICES=1 python -c "
import torch, sys
sys.path.insert(0, 'python')
from attention import attention
from benchmark import measure_mfu
B, H, N, D = 8, 12, 1024, 64
q = torch.randn(B,H,N,D, device='cuda', dtype=torch.bfloat16)
k = torch.randn(B,H,N,D, device='cuda', dtype=torch.bfloat16)
v = torch.randn(B,H,N,D, device='cuda', dtype=torch.bfloat16)
m = measure_mfu(lambda: attention(q,k,v), B,H,N,D)
print(f'latency={m[\"latency_ms\"]:.3f}ms  MFU={m[\"mfu_percent\"]:.1f}%')
print(f'vs FlashInfer(0.311ms): {0.311/m[\"latency_ms\"]:.3f}x')
"
```

**Step 4 — NCU Profile（每轮必跑，无例外）**
```bash
CUDA_VISIBLE_DEVICES=1 ncu \
  --set full --target-processes all \
  -o ncu_reports/our_kernel_n1024.ncu-rep \
  python ncu_reports/profile_our_kernel_ncu.py
```

**Step 5 — NCU 指标解析（以下 7 项必须全部填入 session log，null 不可接受）**
```bash
CUDA_VISIBLE_DEVICES=1 ncu \
  --import ncu_reports/our_kernel_n1024.ncu-rep \
  --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__cycles_active.avg.pct_of_peak_sustained_elapsed,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    launch__registers_per_thread,\
    launch__shared_mem_per_block_static,\
    l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
    lts__throughput.avg.pct_of_peak_sustained_elapsed \
  --print-summary per-kernel 2>&1 | head -60
```
必须记录的 7 项（**全部为 null 视为 evaluator 未完成工作**）：
| 字段 | NCU 指标 | 说明 |
|------|----------|------|
| `compute_throughput_pct` | `sm__throughput` | 核心：< 40% 说明不在 compute bound |
| `sm_active_cycles_pct` | `sm__cycles_active` | SM 活跃率 |
| `occupancy_pct` | `sm__warps_active` | 实际 occupancy |
| `registers_per_thread` | `launch__registers_per_thread` | 寄存器压力 |
| `smem_kb` | `launch__shared_mem_per_block_static` | smem 使用 |
| `l1tex_throughput_pct` | `l1tex__throughput` | L1/纹理缓存带宽占用 |
| `l2_throughput_pct` | `lts__throughput` | L2 带宽占用 |

**Step 6 — Go/No-Go 判断（必须明确写入 notes）**
- **退化（latency 变差）**：必须写明具体原因，不能只写"regression"。
- **平台期（连续2轮改善 < 5%）**：必须在 ROUND_PLAN.json 中标记"换方向"。
- **根本性障碍**（如内存布局不兼容）：必须写"放弃此方向 + 原因"，不能继续 debug 同一条路。
- **compute_throughput < 40% 且 occupancy < 25%**：说明 WMMA 已到天花板，planner 必须切换 WGMMA。

**Step 7 — 写入 session log**
`ncu_metrics` 中 7 项均不得为 null，否则视为 evaluator 工作不完整，本轮结果无效。
