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

---

## 经验教训（反复踩坑记录）

### WGMMA 累加器布局陷阱
- **`rs` form**（A 从寄存器）：每个 thread 的 d[] 覆盖 **9 行** m-values（r0..r0+8，含重复），与在线 softmax 的「每 thread 只跟踪 r0/r1 两行的 l/m」完全不兼容。
- **`ss` form + 128B swizzle**：每个 thread 覆盖 **8 行 × 4 n-values**，同样不兼容。
- **结论**：WGMMA 实现 flash attention 需要 FlashAttention-3 级别的完整架构（warp 专用化 + TMA + warpgroup 级 online softmax），**不能直接把 WMMA 替换成 WGMMA**。

### cp.async 对齐要求
- `cp.async.cg.shared.global [smem], [global], 16` 要求目标 smem 地址 **16 字节对齐**。
- smem 数组行宽必须是 16 的倍数字节：HD=64 → 64×2=128B ✓；HD+2=66 → 66×2=132B ✗（不能用于 cp.async）。
- **不能用**带 +2 padding 的 K/V smem（132B/row）做 cp.async。

### cp.async 双缓冲 + 单缓冲的正确管道流程
- **单缓冲**：无法在加载下一块 K/V 的同时使用当前块（同一 smem 数组），实际上退化为「计算完再等待」，无重叠收益。
- **双缓冲**要求额外 smem，需确认 smem 总量 < 48KB（静态 smem 默认上限）。

### TILE_KV 扫描结果（B=8, H=12, N=1024, D=64）
| TILE_KV | latency | qk_acc frags | 说明 |
|---------|---------|--------------|------|
| 16 | 0.566ms | [1] | 迭代次数过多，overhead 大 |
| 32 | 0.491ms | [2] | 较优 |
| **48** | **0.472ms** | **[3]** | **最优（当前最佳）** |
| 64 | 0.574ms | [4] | 寄存器过多（168→高压力）|

→ **TILE_KV=48 是 WMMA 方案的最优点**，smem ≈27KB，regs ≈100/thread，occupancy ≈23%。

### __launch_bounds__ 使用限制
- `(128, N)` 中 N 越大 → 寄存器上限越低 → 可能触发 register spilling → 反而变慢。
- 经验：`(128, 3)` 无效（不降 regs），`(128, 4)` 触发 spill（0.716ms），`(128, 5)` 轻微 spill（0.478ms）。
- TILE_KV=48 时直接不加 launch_bounds 效果最好（0.472ms）。

### WMMA 性能上限
- WMMA（legacy `mma.sync`）在 SM90 上不走 4th-gen Tensor Core 原生路径，compute throughput 卡在 ~38%，occupancy ~23%，无法超越 FlashInfer 0.311ms。
- M3（<0.311ms）必须用 WGMMA + TMA + warp 专用化架构。

### smem 超过 48KB 默认上限
- 静态 `__shared__` 如果超过 49152 字节（48KB），kernel launch 会报 `cudaErrorInvalidArgument`。
- 双缓冲 K/V（Round 3-4 尝试）：Q[64][66]+K[2][64][66]+V[2][64][66]+S[64][66] = 49664B > 48KB → 必须减小或动态申请。

### TMA + WGMMA B128 根本不兼容（Round 12-13 终极诊断）
- **TMA 正确协议**（CUDA 12.6 H800）：用 `mbarrier.arrive.expect_tx`（既 arrive 又 expect TX），TMA 指令需 `.tile`，smem 32-bit，global 64-bit
- **B128 WGMMA K-inner stride W=8 uint128_t = 8 完整行** → 对 [TKV][HD] 矩阵不兼容（同行相邻列 != 8 行间距）
- TMA 加载后 B128 描述符仍错（实测：所有 j-group 读 rows 7-8，差值 512/j 变成 512+j*8=wrong）
- **结论**：WGMMA flash attention 必须 FA3 完整架构（warp 专用化 + TMA interleaved tile + warpgroup softmax）
