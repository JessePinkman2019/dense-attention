#include "attention.h"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <float.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// ================================================================
// Round 14: WGMMA m64n64k16 BF16 + cp.async double-buffer KV prefetch
//
// Build on Round 11 WGMMA ss-form + 128B swizzle smem.
// Key change: K_smem and V_smem are now double-buffered ([2][TKV][HD]).
// cp.async 128-bit loads replace scalar K/V global->smem stores.
// While computing QK+softmax+AV for tile kv, prefetch tile kv+1 async.
//
// smem layout: Q(8KB) + K×2(16KB) + V×2(16KB) + S(8KB) = 48KB (limit)
// ================================================================

static constexpr int TQ  = 64;
static constexpr int TKV = 64;
static constexpr int HD  = 64;
static constexpr int BLK = 128;  // 1 warpgroup

#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#endif

// ---- 128B swizzle helper ----
__device__ __forceinline__ int swiz(int row, int col) {
    return ((col >> 3) ^ (row & 7)) << 3 | (col & 7);
}

// ---- cp.async 128-bit load (16 bytes) ----
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    unsigned dst32 = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(dst32), "l"((unsigned long long)src) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}

// ---- WGMMA descriptor helpers ----
__device__ __forceinline__ uint64_t make_desc_base(const __nv_bfloat16* pat) {
    uint32_t pa = __cvta_generic_to_shared(pat);
    uint32_t base_off = (pa % 1024 == 0) ? 0u : ((pa >> 7) & 7u);
    return ((uint64_t)64u << 32) | ((uint64_t)base_off << 49) | ((uint64_t)1u << 62);
}

__device__ __forceinline__ uint64_t set_addr(uint64_t base, const __nv_bfloat16* ptr) {
    uint32_t lo = (uint32_t)base;
    lo += __cvta_generic_to_shared(ptr) >> 4;
    return (base & 0xFFFFFFFF00000000ULL) | lo;
}

// ---- WGMMA m64n64k16 BF16, transA=0, transB=1 (for Q×K^T) ----
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
    const int warp_id = threadIdx.x >> 5;
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

    // 48KB smem: Q(8KB) + K[2](16KB) + V[2](16KB) + S(8KB)
    // All arrays start at 1024B boundaries so make_desc_base gives base_off=0
    __shared__ __align__(1024) __nv_bfloat16 Q_smem[TQ ][HD ];         // 8KB
    __shared__ __align__(1024) __nv_bfloat16 K_smem[2][TKV][HD ];      // 16KB
    __shared__ __align__(1024) __nv_bfloat16 V_smem[2][TKV][HD ];      // 16KB
    __shared__                 __nv_bfloat16 S_smem[TQ ][TKV];         // 8KB

    float o_acc[8][2][2];
    #pragma unroll
    for (int j = 0; j < 8; j++)
        o_acc[j][0][0] = o_acc[j][0][1] = o_acc[j][1][0] = o_acc[j][1][1] = 0.f;

    float m0 = -FLT_MAX, m1 = -FLT_MAX;
    float l0 = 0.f, l1 = 0.f;

    const int row0_local = warp_id * 16 + (lane >> 2);
    const int row1_local = row0_local + 8;

    // ---- Load Q into smem with 128B swizzle ----
    #pragma unroll 2
    for (int idx = threadIdx.x; idx < TQ * HD; idx += BLK) {
        int r = idx / HD, c = idx % HD;
        int g = q_start + r;
        Q_smem[r][swiz(r, c)] = (g < N) ? q_ptr[(long)g * HD + c] : __float2bfloat16(0.f);
    }
    __syncthreads();

    // Build descriptor bases
    uint64_t q_base   = make_desc_base(&Q_smem[0][0]);
    uint64_t k_base0  = make_desc_base(&K_smem[0][0][0]);
    uint64_t k_base1  = make_desc_base(&K_smem[1][0][0]);
    uint64_t v_base0  = make_desc_base(&V_smem[0][0][0]);
    uint64_t v_base1  = make_desc_base(&V_smem[1][0][0]);
    uint64_t s_base   = make_desc_base(&S_smem[0][0]);

    const int num_kv = CEIL_DIV(N, TKV);

    // ---- Prologue: prefetch kv=0 into buf=0 ----
    {
        const int kv_len0 = min(TKV, N);
        // TKV*HD/8 = 64*64/8 = 512 chunks of 16 bytes each
        for (int idx = threadIdx.x; idx < TKV * HD / 8; idx += BLK) {
            int r       = idx / (HD / 8);
            int c_group = idx % (HD / 8);       // which 8-BF16 group (0..7)
            int phys    = swiz(r, c_group * 8); // physical col start
            if (r < kv_len0) {
                long off = (long)r * HD + c_group * 8;
                cp_async16(&K_smem[0][r][phys], k_ptr + off);
                cp_async16(&V_smem[0][r][phys], v_ptr + off);
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    K_smem[0][r][phys + j] = __float2bfloat16(0.f);
                    V_smem[0][r][phys + j] = __float2bfloat16(0.f);
                }
            }
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();
    }

    for (int kv = 0; kv < num_kv; ++kv) {
        const int buf      = kv & 1;
        const int next_buf = 1 - buf;
        const int kv_start = kv * TKV;
        const int kv_len   = min(TKV, N - kv_start);

        // ---- Issue async prefetch for kv+1 into next_buf ----
        // (overlaps with QK WGMMA + online softmax below)
        if (kv + 1 < num_kv) {
            const int nxt_start = (kv + 1) * TKV;
            const int nxt_len   = min(TKV, N - nxt_start);
            for (int idx = threadIdx.x; idx < TKV * HD / 8; idx += BLK) {
                int r       = idx / (HD / 8);
                int c_group = idx % (HD / 8);
                int phys    = swiz(r, c_group * 8);
                if (r < nxt_len) {
                    long off = (long)(nxt_start + r) * HD + c_group * 8;
                    cp_async16(&K_smem[next_buf][r][phys], k_ptr + off);
                    cp_async16(&V_smem[next_buf][r][phys], v_ptr + off);
                } else {
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        K_smem[next_buf][r][phys + j] = __float2bfloat16(0.f);
                        V_smem[next_buf][r][phys + j] = __float2bfloat16(0.f);
                    }
                }
            }
            cp_async_commit();
        }

        // ---- QK WGMMA: qk_acc = Q_smem × K_smem[buf]^T (transB=1) ----
        float qk_acc[8][2][2];
        uint64_t k_base = (buf == 0) ? k_base0 : k_base1;
        asm volatile("wgmma.fence.sync.aligned;\n");
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            uint64_t dA = set_addr(q_base, &Q_smem[0][kt * 16]);
            uint64_t dB = set_addr(k_base, &K_smem[buf][0][kt * 16]);
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

        // ---- Mask padding ----
        if (kv_len < TKV) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int c0 = j * 8 + (lane & 3) * 2;
                if (c0     >= kv_len) { qk_acc[j][0][0] = qk_acc[j][1][0] = -FLT_MAX; }
                if (c0 + 1 >= kv_len) { qk_acc[j][0][1] = qk_acc[j][1][1] = -FLT_MAX; }
            }
        }

        // ---- Online softmax ----
        float pm0 = -FLT_MAX, pm1 = -FLT_MAX;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            pm0 = fmaxf(pm0, fmaxf(qk_acc[j][0][0], qk_acc[j][0][1]));
            pm1 = fmaxf(pm1, fmaxf(qk_acc[j][1][0], qk_acc[j][1][1]));
        }
        pm0 = fmaxf(pm0, __shfl_xor_sync(0xffffffff, pm0, 1));
        pm0 = fmaxf(pm0, __shfl_xor_sync(0xffffffff, pm0, 2));
        pm1 = fmaxf(pm1, __shfl_xor_sync(0xffffffff, pm1, 1));
        pm1 = fmaxf(pm1, __shfl_xor_sync(0xffffffff, pm1, 2));

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
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 1);
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 1);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
        l0 += sum0;
        l1 += sum1;

        // ---- Wait for kv+1 prefetch to complete, then sync for S_smem ----
        if (kv + 1 < num_kv) {
            cp_async_wait_all();
        }
        __syncthreads();  // S_smem visible + K/V[next_buf] ready

        // ---- AV WGMMA: o_acc += S_smem × V_smem[buf] (transB=0) ----
        uint64_t v_base = (buf == 0) ? v_base0 : v_base1;
        asm volatile("wgmma.fence.sync.aligned;\n");
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            uint64_t dA = set_addr(s_base, &S_smem[0][kt * 16]);
            uint64_t dB = set_addr(v_base, &V_smem[buf][kt * 16][0]);
            wgmma_av(o_acc, dA, dB);
        }
        asm volatile("wgmma.commit_group.sync.aligned;\n");
        asm volatile("wgmma.wait_group.sync.aligned 0;\n");

        __syncthreads();
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
    m.def("forward", &attn_forward, "Dense attention BF16 (Round 14: WGMMA + cp.async double-buffer KV)");
}
