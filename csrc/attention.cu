#include "attention.h"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <float.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// ================================================================
// Round 11: WGMMA m64n64k16 BF16 with 128B swizzle smem
//
// Key design:
//   - 128 threads = 1 warpgroup (warp 0-3)
//   - TILE_Q=64, TILE_KV=64, HD=64
//   - smem: 4 × [64×64 BF16] = 32KB (< 48KB)
//   - WGMMA acc per thread: float [8][2][2] = 32 fp32
//
// WGMMA m64n64k16 accumulator layout (per thread):
//   warp_id = tid/32, lane = tid%32
//   row_0  = warp_id*16 + lane/4      (maps to rows 0..63, 4 threads per row)
//   row_1  = row_0 + 8
//   acc[j][rb][cb]: row = row_0 + rb*8, col = j*8 + (lane%4)*2 + cb
//
// Online softmax:
//   Each thread tracks m[2] and l[2] (one per row).
//   Row max reduce: __shfl_xor_sync with masks 1 and 2 (within quad).
// ================================================================

static constexpr int TQ  = 64;
static constexpr int TKV = 64;
static constexpr int HD  = 64;
static constexpr int BLK = 128;  // 1 warpgroup

#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#endif

// ---- 128B swizzle helper ----
// Physical column for logical [row][col] with 128B swizzle
// (col group = col/8, XOR'd with row%8; then add within-group offset)
__device__ __forceinline__ int swiz(int row, int col) {
    return ((col >> 3) ^ (row & 7)) << 3 | (col & 7);
}

// ---- WGMMA descriptor helpers ----
// Build base descriptor (addr=0) for a [64][64] BF16 smem with 128B swizzle.
// dimMNOffset=64 encodes the 8-row group stride (8*128B = 1024B; 1024/16=64).
// baseOffset depends on array's 1024B-alignment (0 for our 8KB-aligned arrays).
__device__ __forceinline__ uint64_t make_desc_base(const __nv_bfloat16* pat) {
    uint32_t pa = __cvta_generic_to_shared(pat);
    uint32_t base_off = (pa % 1024 == 0) ? 0u : ((pa >> 7) & 7u);
    // bits 32-47: dimMNOffset=64, bits 49-51: baseOffset, bits 62-63: swizzle=1 (128B)
    return ((uint64_t)64u << 32) | ((uint64_t)base_off << 49) | ((uint64_t)1u << 62);
}

// Replace addr field (bits 0-15) with smem pointer >> 4
__device__ __forceinline__ uint64_t set_addr(uint64_t base, const __nv_bfloat16* ptr) {
    uint32_t lo = (uint32_t)base;
    lo += __cvta_generic_to_shared(ptr) >> 4;
    return (base & 0xFFFFFFFF00000000ULL) | lo;
}

// ---- WGMMA m64n64k16 BF16, transA=0, transB=1 (for Q×K^T) ----
// scale_d is an immediate: 0=reset acc, 1=accumulate
#define WGMMA_QK_ASM(SCALE_D)                                                     \
    asm volatile(                                                                  \
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\n"                 \
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"                \
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},\n"    \
        "%32,\n%33,\n" #SCALE_D ", 1, 1, 0, 1;\n"                               \
        :"+f"(acc[0][0][0]),"+f"(acc[0][0][1]),"+f"(acc[0][1][0]),"+f"(acc[0][1][1]), \
         "+f"(acc[1][0][0]),"+f"(acc[1][0][1]),"+f"(acc[1][1][0]),"+f"(acc[1][1][1]), \
         "+f"(acc[2][0][0]),"+f"(acc[2][0][1]),"+f"(acc[2][1][0]),"+f"(acc[2][1][1]), \
         "+f"(acc[3][0][0]),"+f"(acc[3][0][1]),"+f"(acc[3][1][0]),"+f"(acc[3][1][1]), \
         "+f"(acc[4][0][0]),"+f"(acc[4][0][1]),"+f"(acc[4][1][0]),"+f"(acc[4][1][1]), \
         "+f"(acc[5][0][0]),"+f"(acc[5][0][1]),"+f"(acc[5][1][0]),"+f"(acc[5][1][1]), \
         "+f"(acc[6][0][0]),"+f"(acc[6][0][1]),"+f"(acc[6][1][0]),"+f"(acc[6][1][1]), \
         "+f"(acc[7][0][0]),"+f"(acc[7][0][1]),"+f"(acc[7][1][0]),"+f"(acc[7][1][1]) \
        :"l"(dA),"l"(dB))

__device__ __forceinline__
void wgmma_qk(float acc[8][2][2], uint64_t dA, uint64_t dB, bool hasVal) {
    if (hasVal) { WGMMA_QK_ASM(1); }
    else         { WGMMA_QK_ASM(0); }
}
#undef WGMMA_QK_ASM

// ---- WGMMA m64n64k16 BF16, transA=0, transB=0 (for S×V) ----
// Always accumulates (scale_d=1)
__device__ __forceinline__
void wgmma_av(float acc[8][2][2], uint64_t dA, uint64_t dB) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},\n"
        "%32,\n%33,\n1, 1, 1, 0, 0;\n"
        :"+f"(acc[0][0][0]),"+f"(acc[0][0][1]),"+f"(acc[0][1][0]),"+f"(acc[0][1][1]),
         "+f"(acc[1][0][0]),"+f"(acc[1][0][1]),"+f"(acc[1][1][0]),"+f"(acc[1][1][1]),
         "+f"(acc[2][0][0]),"+f"(acc[2][0][1]),"+f"(acc[2][1][0]),"+f"(acc[2][1][1]),
         "+f"(acc[3][0][0]),"+f"(acc[3][0][1]),"+f"(acc[3][1][0]),"+f"(acc[3][1][1]),
         "+f"(acc[4][0][0]),"+f"(acc[4][0][1]),"+f"(acc[4][1][0]),"+f"(acc[4][1][1]),
         "+f"(acc[5][0][0]),"+f"(acc[5][0][1]),"+f"(acc[5][1][0]),"+f"(acc[5][1][1]),
         "+f"(acc[6][0][0]),"+f"(acc[6][0][1]),"+f"(acc[6][1][0]),"+f"(acc[6][1][1]),
         "+f"(acc[7][0][0]),"+f"(acc[7][0][1]),"+f"(acc[7][1][0]),"+f"(acc[7][1][1])
        :"l"(dA),"l"(dB));
}

// ---- Main kernel ----
__global__ void flash_attn_wgmma_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
          __nv_bfloat16* __restrict__ O,
          float*         __restrict__ LSE,
    int N, int H, float scale)
{
    const int warp_id = threadIdx.x >> 5;   // 0..3
    const int lane    = threadIdx.x & 31;
    const int i_h     = blockIdx.y;
    const int i_b     = blockIdx.z;
    const int q_start = (int)blockIdx.x * TQ;

    const long stride_H = (long)N * HD;
    const long stride_B = (long)H * stride_H;

    const __nv_bfloat16* q_ptr = Q + i_b * stride_B + i_h * stride_H;
    const __nv_bfloat16* k_ptr = K + i_b * stride_B + i_h * stride_H;
    const __nv_bfloat16* v_ptr = V + i_b * stride_B + i_h * stride_H;
          __nv_bfloat16* o_ptr = O + i_b * stride_B + i_h * stride_H;

    // 32KB smem; aligned to 1024B so baseOffset=0 for all arrays
    __shared__ __align__(1024) __nv_bfloat16 Q_smem[TQ ][HD ];   // 8KB
    __shared__                 __nv_bfloat16 K_smem[TKV][HD ];   // 8KB (starts at offset 8KB = 8×1024)
    __shared__                 __nv_bfloat16 V_smem[TKV][HD ];   // 8KB
    __shared__                 __nv_bfloat16 S_smem[TQ ][TKV];   // 8KB

    // Output accumulator (persists across KV tiles)
    float o_acc[8][2][2];
    #pragma unroll
    for (int j = 0; j < 8; j++)
        o_acc[j][0][0] = o_acc[j][0][1] = o_acc[j][1][0] = o_acc[j][1][1] = 0.f;

    float m0 = -FLT_MAX, m1 = -FLT_MAX;
    float l0 = 0.f, l1 = 0.f;

    // Per-thread row positions in the WGMMA accumulator
    const int row0_local = warp_id * 16 + (lane >> 2);   // row_0 within 64-row tile
    const int row1_local = row0_local + 8;

    // ---- Load Q into smem with 128B swizzle ----
    #pragma unroll 2
    for (int idx = threadIdx.x; idx < TQ * HD; idx += BLK) {
        int r = idx / HD, c = idx % HD;
        int g = q_start + r;
        Q_smem[r][swiz(r, c)] = (g < N) ? q_ptr[(long)g * HD + c] : __float2bfloat16(0.f);
    }
    __syncthreads();

    // Build descriptor bases (addr=0; actual ptr added via set_addr)
    uint64_t q_base = make_desc_base(&Q_smem[0][0]);
    uint64_t k_base = make_desc_base(&K_smem[0][0]);
    uint64_t v_base = make_desc_base(&V_smem[0][0]);
    uint64_t s_base = make_desc_base(&S_smem[0][0]);

    const int num_kv = CEIL_DIV(N, TKV);
    for (int kv = 0; kv < num_kv; ++kv) {
        const int kv_start = kv * TKV;
        const int kv_len   = min(TKV, N - kv_start);

        // ---- Load K and V into smem with 128B swizzle ----
        #pragma unroll 2
        for (int idx = threadIdx.x; idx < TKV * HD; idx += BLK) {
            int r = idx / HD, c = idx % HD;
            int g = kv_start + r;
            __nv_bfloat16 kv = (r < kv_len) ? k_ptr[(long)g * HD + c] : __float2bfloat16(0.f);
            __nv_bfloat16 vv = (r < kv_len) ? v_ptr[(long)g * HD + c] : __float2bfloat16(0.f);
            K_smem[r][swiz(r, c)] = kv;
            V_smem[r][swiz(r, c)] = vv;
        }
        __syncthreads();

        // ---- QK WGMMA: qk_acc = Q_smem × K_smem^T (transB=1) ----
        // k-loop over HD=64 in steps of 16
        float qk_acc[8][2][2];
        asm volatile("wgmma.fence.sync.aligned;\n");
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            uint64_t dA = set_addr(q_base, &Q_smem[0][kt * 16]);
            uint64_t dB = set_addr(k_base, &K_smem[0][kt * 16]);
            wgmma_qk(qk_acc, dA, dB, kt > 0 ? 1 : 0);
        }
        asm volatile("wgmma.commit_group.sync.aligned;\n");
        asm volatile("wgmma.wait_group.sync.aligned 0;\n");

        // ---- Scale ----
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            qk_acc[j][0][0] *= scale; qk_acc[j][0][1] *= scale;
            qk_acc[j][1][0] *= scale; qk_acc[j][1][1] *= scale;
        }

        // ---- Mask padding (last tile) ----
        if (kv_len < TKV) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int c0 = j * 8 + (lane & 3) * 2;
                if (c0     >= kv_len) { qk_acc[j][0][0] = qk_acc[j][1][0] = -FLT_MAX; }
                if (c0 + 1 >= kv_len) { qk_acc[j][0][1] = qk_acc[j][1][1] = -FLT_MAX; }
            }
        }

        // ---- Online softmax: row-max, rescale, exp, row-sum ----

        // 1. Partial row max within thread
        float pm0 = -FLT_MAX, pm1 = -FLT_MAX;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            pm0 = fmaxf(pm0, fmaxf(qk_acc[j][0][0], qk_acc[j][0][1]));
            pm1 = fmaxf(pm1, fmaxf(qk_acc[j][1][0], qk_acc[j][1][1]));
        }

        // 2. Reduce within quad (xor 1 then xor 2 → max across all 4 threads in quad)
        pm0 = fmaxf(pm0, __shfl_xor_sync(0xffffffff, pm0, 1));
        pm0 = fmaxf(pm0, __shfl_xor_sync(0xffffffff, pm0, 2));
        pm1 = fmaxf(pm1, __shfl_xor_sync(0xffffffff, pm1, 1));
        pm1 = fmaxf(pm1, __shfl_xor_sync(0xffffffff, pm1, 2));

        // 3. Update running max, compute rescale factors for o_acc and l
        const float m0_new = fmaxf(m0, pm0);
        const float m1_new = fmaxf(m1, pm1);
        const float a0s = __expf(m0 - m0_new);
        const float a1s = __expf(m1 - m1_new);

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            o_acc[j][0][0] *= a0s; o_acc[j][0][1] *= a0s;
            o_acc[j][1][0] *= a1s; o_acc[j][1][1] *= a1s;
        }
        l0 *= a0s;  m0 = m0_new;
        l1 *= a1s;  m1 = m1_new;

        // 4. Compute exp(qk - m), write P to S_smem with 128B swizzle, accumulate sum
        float sum0 = 0.f, sum1 = 0.f;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            #pragma unroll
            for (int cb = 0; cb < 2; cb++) {
                const float p0 = __expf(qk_acc[j][0][cb] - m0);
                const float p1 = __expf(qk_acc[j][1][cb] - m1);
                sum0 += p0;
                sum1 += p1;
                const int log_col = j * 8 + (lane & 3) * 2 + cb;
                S_smem[row0_local][swiz(row0_local, log_col)] = __float2bfloat16(p0);
                S_smem[row1_local][swiz(row1_local, log_col)] = __float2bfloat16(p1);
            }
        }

        // 5. Reduce sum within quad (same XOR pattern as max)
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 1);
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 1);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
        l0 += sum0;
        l1 += sum1;

        __syncthreads();  // Ensure S_smem is fully written before WGMMA reads it

        // ---- AV WGMMA: o_acc += S_smem × V_smem (transB=0) ----
        // k-loop over TKV=64 in steps of 16
        asm volatile("wgmma.fence.sync.aligned;\n");
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            uint64_t dA = set_addr(s_base, &S_smem[0][kt * 16]);
            uint64_t dB = set_addr(v_base, &V_smem[kt * 16][0]);
            wgmma_av(o_acc, dA, dB);
        }
        asm volatile("wgmma.commit_group.sync.aligned;\n");
        asm volatile("wgmma.wait_group.sync.aligned 0;\n");

        __syncthreads();  // Allow K/V smem to be reused next iteration
    }

    // ---- Normalize output ----
    const float inv_l0 = 1.f / l0;
    const float inv_l1 = 1.f / l1;

    const int row0_g = q_start + row0_local;
    const int row1_g = q_start + row1_local;

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int c0 = j * 8 + (lane & 3) * 2;
        if (row0_g < N) {
            o_ptr[(long)row0_g * HD + c0    ] = __float2bfloat16(o_acc[j][0][0] * inv_l0);
            o_ptr[(long)row0_g * HD + c0 + 1] = __float2bfloat16(o_acc[j][0][1] * inv_l0);
        }
        if (row1_g < N) {
            o_ptr[(long)row1_g * HD + c0    ] = __float2bfloat16(o_acc[j][1][0] * inv_l1);
            o_ptr[(long)row1_g * HD + c0 + 1] = __float2bfloat16(o_acc[j][1][1] * inv_l1);
        }
    }

    if (LSE && (lane & 3) == 0) {
        const long lse_base = (long)i_b * H * N + i_h * N;
        if (row0_g < N) LSE[lse_base + row0_g] = m0 + __logf(l0);
        if (row1_g < N) LSE[lse_base + row1_g] = m1 + __logf(l1);
    }
}

void attention_fwd_bf16(const AttentionParams& p, cudaStream_t stream)
{
    TORCH_CHECK(p.head_dim == HD,
                "head_dim must be ", HD, " for this kernel, got ", p.head_dim);
    const dim3 grid(CEIL_DIV(p.seq_len, TQ), p.num_heads, p.batch_size);
    flash_attn_wgmma_kernel<<<grid, dim3(BLK), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(p.q),
        reinterpret_cast<const __nv_bfloat16*>(p.k),
        reinterpret_cast<const __nv_bfloat16*>(p.v),
        reinterpret_cast<      __nv_bfloat16*>(p.out),
        p.lse,
        p.seq_len, p.num_heads, p.scale);
}

void attention_fwd_fp16(const AttentionParams& /*p*/, cudaStream_t /*stream*/)
{
    TORCH_CHECK(false, "FP16 not implemented");
}

void attn_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor out, torch::Tensor lse,
    float scale, bool is_training)
{
    TORCH_CHECK(q.is_cuda() && q.is_contiguous(), "q must be contiguous CUDA tensor");
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "only bfloat16 supported");

    AttentionParams params;
    params.q           = q.data_ptr();
    params.k           = k.data_ptr();
    params.v           = v.data_ptr();
    params.out         = out.data_ptr();
    params.lse         = (is_training && lse.defined()) ? lse.data_ptr<float>() : nullptr;
    params.batch_size  = q.size(0);
    params.num_heads   = q.size(1);
    params.seq_len     = q.size(2);
    params.head_dim    = q.size(3);
    params.scale       = scale;
    params.is_training = is_training;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    attention_fwd_bf16(params, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attn_forward, "Dense attention BF16 (Round 11: WGMMA m64n64k16 + 128B swizzle)");
}
