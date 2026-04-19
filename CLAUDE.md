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

三个角色均以**独立 Agent** 方式启动，每个 Agent 只读持久化文件（session log / ROUND_PLAN.json / attention.cu），不依赖对话历史：

| 角色 | 触发方式 | 输入 | 输出 |
|------|----------|------|------|
| `/planner` | skill | `optimization_session.json`（最近3轮） | `ROUND_PLAN.json` |
| `/generator` | skill | `ROUND_PLAN.json` + `csrc/attention.cu` | 新 `attention.cu` + git commit |
| `/evaluator` | skill | 最新 git commit | `optimization_session.json` 追加一轮 + Memory 更新 |

**顺序**：planner → generator → evaluator → planner → …（严格串行，不并行）

> **上下文节省关键**：每次只开一个角色 Agent，完成后关闭。下一角色读持久化文件，不看聊天记录。

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
