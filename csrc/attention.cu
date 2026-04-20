// ================================================================
// Round 17: WGMMA m64n64k16 SS-form BF16 Dense Attention
//
// Architecture: H800 (sm_90a), 128 threads = 1 warpgroup (4 warps)
// Tile: TILE_Q=64, TILE_KV=64, D=64
// SMEM: 128B-swizzled layout for WGMMA descriptor compatibility
// Online softmax with intra-warp reduction
//
// CORRECT Accumulator layout (verified by S=I, V=col_indices test):
//   For thread t: lane l=t%32, warp w=t/32
//   For register v=0..31: j=v/4 (0..7), r=(v/2)%2 (0..1), c=v%2 (0..1)
//   M_dim (row of output matrix) = (l >> 2) + r * 8 + w * 16  [0..63]
//   N_dim (col of output matrix) = (l & 3) * 2 + c + j * 8    [0..63]
//
// For QK^T: M = query position, N = key position
// For AV:   M = query position, N = head_dim position
//
// Each thread owns 2 M values (r=0,1) and 16 N values (j=0..7, c=0,1)
// Softmax reduces over N (keys): local max/sum over j,c, then XOR 1,2
// No inter-warp reduction needed: each (warp, r) covers unique M range
// ================================================================

#include "attention.h"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <float.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

static constexpr int TILE_Q   = 64;
static constexpr int TILE_KV  = 64;
static constexpr int HD       = 64;
static constexpr int BLK      = 128;  // 1 warpgroup = 4 warps x 32 threads
static constexpr int NUM_WARPS = 4;

// Number of k-steps for D=64: 64/16 = 4
static constexpr int K_STEPS  = 4;

#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#endif

// ================================================================
// 128B Swizzle for WGMMA smem descriptors
// Swizzle<3,4,3> from CUTLASS: XOR bits [6:4] of byte-offset with bits [9:7]
// For bf16 (2 bytes/elem), 64 elems/row = 128 bytes/row
// ================================================================
__device__ __forceinline__ int swiz(int row, int col) {
    return col ^ ((row & 7) << 3);
}

// ================================================================
// WGMMA smem descriptor construction
// ================================================================
__device__ __forceinline__ uint64_t make_smem_desc(const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = 0;
    desc |= (uint64_t)((addr & 0x3FFFFu) >> 4);
    desc |= (uint64_t)(((uint32_t)16 >> 4) & 0x3FFFu) << 16;
    desc |= (uint64_t)(((uint32_t)1024 >> 4) & 0x3FFFu) << 32;
    desc |= 1ULL << 62;
    return desc;
}

// Advance descriptor for K-major operand: 16 bf16 columns = 32 bytes -> desc += 2
__device__ __forceinline__ uint64_t desc_advance_k_major(uint64_t desc) {
    return desc + 2;
}

// ================================================================
// warpgroup_fence_operand
// ================================================================
__device__ __forceinline__ void wgmma_fence_reg(float& reg) {
    asm volatile("" : "+f"(reg) :: "memory");
}

// ================================================================
// WGMMA m64n64k16 bf16 SS-form: both K-major (trans=0,0)
// Computes C[m][n] = sum_k A[m][k] * B[n][k]  (i.e. A * B^T)
// ================================================================
__device__ __forceinline__ void wgmma_ss_KK(
    float acc[32],
    uint64_t desc_a,
    uint64_t desc_b,
    int scale_d)
{
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "%32, %33, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(acc[0]),  "+f"(acc[1]),  "+f"(acc[2]),  "+f"(acc[3]),
          "+f"(acc[4]),  "+f"(acc[5]),  "+f"(acc[6]),  "+f"(acc[7]),
          "+f"(acc[8]),  "+f"(acc[9]),  "+f"(acc[10]), "+f"(acc[11]),
          "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]),
          "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]),
          "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]),
          "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]),
          "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31])
        : "l"(desc_a), "l"(desc_b), "r"(scale_d)
    );
}

// ================================================================
// Main WGMMA Flash Attention Kernel
//
// Accumulator register v -> (M_dim, N_dim):
//   j = v / 4, r = (v / 2) % 2, c = v % 2
//   M_dim = (lane >> 2) + r * 8 + warp_id * 16   (query row)
//   N_dim = (lane & 3) * 2 + c + j * 8            (key col or head_dim col)
//
// For QK: A=Q, B=K, C=QK^T. M=query, N=key. Both K-major.
// For AV: A=S, B=V_T. M=query, N=head_dim. Both K-major.
//   V_smem stores V transposed: V_smem[head_dim][key_pos]
//   wgmma_ss_KK computes S * V_smem^T = S * V
//
// Softmax:
//   Need to reduce over N (all 64 key columns) for each M (query row).
//   Each thread: local reduction over 16 N values (j=0..7, c=0,1)
//   Intra-warp: XOR 1, 2 to combine 4 lanes with same (lane >> 2)
//   No inter-warp reduction: each (warp_id, r) covers unique M range
//   m_run[2], l_run[2]: indexed by r (0 or 1)
// ================================================================
__global__ void __launch_bounds__(128, 1)
flash_attn_wgmma_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
          __nv_bfloat16* __restrict__ O,
          float*         __restrict__ LSE,
    int N, int H, float scale)
{
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid % 32;
    const int i_h     = blockIdx.y;
    const int i_b     = blockIdx.z;
    const int q_start = blockIdx.x * TILE_Q;

    const long stride_H = (long)N * HD;
    const long stride_B = (long)H * N * HD;

    const __nv_bfloat16* q_ptr = Q + i_b * stride_B + i_h * stride_H;
    const __nv_bfloat16* k_ptr = K + i_b * stride_B + i_h * stride_H;
    const __nv_bfloat16* v_ptr = V + i_b * stride_B + i_h * stride_H;
          __nv_bfloat16* o_ptr = O + i_b * stride_B + i_h * stride_H;

    // Shared memory layout (dynamic):
    //   Q_smem[64*64]: 8192 bytes
    //   K_smem[64*64]: 8192 bytes
    //   V_smem[64*64]: 8192 bytes (V stored transposed)
    //   S_smem[64*64]: 8192 bytes (softmax scores)
    // Total: 32768 bytes
    extern __shared__ char smem_raw[];

    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* K_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 8192);
    __nv_bfloat16* V_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    __nv_bfloat16* S_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 24576);

    // Precompute M_dim for this thread's two r values
    // M_dim[0] = (lane >> 2) + 0 * 8 + warp_id * 16  (r=0)
    // M_dim[1] = (lane >> 2) + 1 * 8 + warp_id * 16  (r=1)
    const int M_base = (lane >> 2) + warp_id * 16;
    // M_dim[r] = M_base + r * 8

    // Output accumulator
    float o_acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) o_acc[i] = 0.0f;

    // Online softmax running state per M value (2 per thread: r=0 and r=1)
    float m_run[2] = {-FLT_MAX, -FLT_MAX};
    float l_run[2] = {0.0f, 0.0f};

    // ============================================================
    // Load Q tile into smem with 128B swizzle
    // ============================================================
    for (int idx = tid; idx < TILE_Q * HD; idx += BLK) {
        int row = idx / HD;
        int col = idx % HD;
        int g = q_start + row;
        __nv_bfloat16 val = (g < N) ? q_ptr[(long)g * HD + col] : __float2bfloat16(0.0f);
        Q_smem[row * HD + swiz(row, col)] = val;
    }
    __syncthreads();

    // ============================================================
    // Main loop over KV tiles
    // ============================================================
    const int num_kv = CEIL_DIV(N, TILE_KV);

    for (int kv = 0; kv < num_kv; ++kv) {
        const int kv_start = kv * TILE_KV;
        const int kv_len   = min(TILE_KV, N - kv_start);

        // Load K and V tiles
        // K: K_smem[key_pos][head_dim], K-major (row = key, col = dim)
        // V: V_smem[head_dim][key_pos], transposed (row = dim, col = key)
        for (int idx = tid; idx < TILE_KV * HD; idx += BLK) {
            int r = idx / HD;
            int c = idx % HD;
            int g = kv_start + r;
            __nv_bfloat16 kval = (r < kv_len) ? k_ptr[(long)g * HD + c] : __float2bfloat16(0.0f);
            __nv_bfloat16 vval = (r < kv_len) ? v_ptr[(long)g * HD + c] : __float2bfloat16(0.0f);
            K_smem[r * HD + swiz(r, c)] = kval;
            V_smem[c * TILE_KV + swiz(c, r)] = vval;  // Transposed
        }
        __syncthreads();

        // ============================================================
        // QK^T via WGMMA m64n64k16 SS-form
        // A=Q (K-major), B=K (K-major) -> C = Q * K^T
        // C[M_dim][N_dim] = QK^T[query][key]
        // ============================================================
        float qk_acc[32];
        {
            uint64_t desc_q = make_smem_desc(&Q_smem[0]);
            uint64_t desc_k = make_smem_desc(&K_smem[0]);

            #pragma unroll
            for (int i = 0; i < 32; i++) wgmma_fence_reg(qk_acc[i]);
            asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

            wgmma_ss_KK(qk_acc, desc_q, desc_k, 0);
            desc_q = desc_advance_k_major(desc_q);
            desc_k = desc_advance_k_major(desc_k);

            #pragma unroll
            for (int kt = 1; kt < K_STEPS; kt++) {
                wgmma_ss_KK(qk_acc, desc_q, desc_k, 1);
                if (kt < K_STEPS - 1) {
                    desc_q = desc_advance_k_major(desc_q);
                    desc_k = desc_advance_k_major(desc_k);
                }
            }

            asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
            asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
            #pragma unroll
            for (int i = 0; i < 32; i++) wgmma_fence_reg(qk_acc[i]);
        }

        // ============================================================
        // Scale QK by softmax temperature
        // ============================================================
        #pragma unroll
        for (int i = 0; i < 32; i++) qk_acc[i] *= scale;

        // ============================================================
        // Masking for partial KV tiles
        // N_dim = key position for QK output
        // ============================================================
        if (kv_len < TILE_KV) {
            #pragma unroll
            for (int v = 0; v < 32; v++) {
                int j = v / 4;
                int c = v % 2;
                int N_dim = (lane & 3) * 2 + c + j * 8;
                if (kv_start + N_dim >= N) {
                    qk_acc[v] = -FLT_MAX;
                }
            }
        }

        // ============================================================
        // Online softmax: compute max per M value (query row)
        //
        // Each thread holds 32 registers, covering 2 M values (r=0,1)
        // and 16 N values (j=0..7, c=0,1).
        //
        // Step 1: Local max over 16 N values per M value
        // Step 2: Intra-warp XOR 1, 2 to combine 4 lanes with same (lane >> 2)
        //         (lanes with same lane>>2 have same M but different N ranges)
        // No inter-warp reduction: each (warp_id, r) has unique M range.
        // ============================================================

        // Local max per r value (over all j, c)
        float local_max[2] = {-FLT_MAX, -FLT_MAX};
        #pragma unroll
        for (int v = 0; v < 32; v++) {
            int r = (v / 2) % 2;
            local_max[r] = fmaxf(local_max[r], qk_acc[v]);
        }

        // Intra-warp reduction: XOR 1, 2 (combine 4 lanes with same lane>>2)
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            local_max[r] = fmaxf(local_max[r], __shfl_xor_sync(0xffffffff, local_max[r], 1));
            local_max[r] = fmaxf(local_max[r], __shfl_xor_sync(0xffffffff, local_max[r], 2));
        }

        // ============================================================
        // Rescale o_acc and l_run with new max
        // ============================================================
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            float m_new = fmaxf(m_run[r], local_max[r]);
            float rescale = __expf(m_run[r] - m_new);

            // Rescale all o_acc registers for this M value (r)
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                #pragma unroll
                for (int c = 0; c < 2; c++) {
                    int v = j * 4 + r * 2 + c;
                    o_acc[v] *= rescale;
                }
            }

            l_run[r] *= rescale;
            m_run[r] = m_new;
        }

        // ============================================================
        // Compute exp(qk - m), write S_smem, accumulate local sum
        // ============================================================
        float local_sum[2] = {0.0f, 0.0f};

        #pragma unroll
        for (int v = 0; v < 32; v++) {
            int j = v / 4;
            int r = (v / 2) % 2;
            int c = v % 2;
            int M_dim = M_base + r * 8;              // query row
            int N_dim = (lane & 3) * 2 + c + j * 8;  // key column

            float exp_val = __expf(qk_acc[v] - m_run[r]);
            local_sum[r] += exp_val;

            // Write S_smem[query_row][key_col] with swizzle
            S_smem[M_dim * HD + swiz(M_dim, N_dim)] = __float2bfloat16(exp_val);
        }

        // Intra-warp sum reduction: XOR 1, 2
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            local_sum[r] += __shfl_xor_sync(0xffffffff, local_sum[r], 1);
            local_sum[r] += __shfl_xor_sync(0xffffffff, local_sum[r], 2);
        }

        // Update running sum
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            l_run[r] += local_sum[r];
        }

        __syncthreads();

        // ============================================================
        // AV WGMMA: S[64x64] * V_smem^T
        // S_smem[query][key], V_smem[dim][key] (transposed V)
        // Both K-major. wgmma_ss_KK computes S * V_smem^T = S * V
        // Accumulates into o_acc (scale_d=1)
        // ============================================================
        {
            uint64_t desc_s = make_smem_desc(&S_smem[0]);
            uint64_t desc_v = make_smem_desc(&V_smem[0]);

            #pragma unroll
            for (int i = 0; i < 32; i++) wgmma_fence_reg(o_acc[i]);
            asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

            #pragma unroll
            for (int kt = 0; kt < K_STEPS; kt++) {
                wgmma_ss_KK(o_acc, desc_s, desc_v, 1);
                if (kt < K_STEPS - 1) {
                    desc_s = desc_advance_k_major(desc_s);
                    desc_v = desc_advance_k_major(desc_v);
                }
            }

            asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
            asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
            #pragma unroll
            for (int i = 0; i < 32; i++) wgmma_fence_reg(o_acc[i]);
        }

        __syncthreads();
    }

    // ============================================================
    // Write output: O[query_row][head_dim] = o_acc / l_run
    // M_dim = query row = (lane >> 2) + r * 8 + warp_id * 16
    // N_dim = head_dim col = (lane & 3) * 2 + c + j * 8
    // ============================================================
    #pragma unroll
    for (int v = 0; v < 32; v++) {
        int j = v / 4;
        int r = (v / 2) % 2;
        int c = v % 2;
        int M_dim = M_base + r * 8;
        int N_dim = (lane & 3) * 2 + c + j * 8;
        int global_row = q_start + M_dim;

        if (global_row < N && N_dim < HD) {
            float inv_l = 1.0f / l_run[r];
            o_ptr[(long)global_row * HD + N_dim] = __float2bfloat16(o_acc[v] * inv_l);
        }
    }

    // ============================================================
    // Write LSE: one value per M_dim (query row)
    // Only one thread per M_dim needs to write.
    // Use (lane & 3) == 0 to deduplicate within the 4-lane group.
    // ============================================================
    if (LSE) {
        const long lse_base = (long)i_b * H * N + i_h * N;
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            int M_dim = M_base + r * 8;
            int global_row = q_start + M_dim;
            if ((lane & 3) == 0 && global_row < N) {
                LSE[lse_base + global_row] = m_run[r] + __logf(l_run[r]);
            }
        }
    }
}

// ============================================================
// Host launch function
// ============================================================
void attention_fwd_bf16(const AttentionParams& p, cudaStream_t stream)
{
    TORCH_CHECK(p.head_dim == HD,
                "head_dim must be ", HD, " for this kernel, got ", p.head_dim);

    const dim3 grid(CEIL_DIV(p.seq_len, TILE_Q), p.num_heads, p.batch_size);
    const dim3 block(BLK);

    // Dynamic shared memory: Q(8KB) + K(8KB) + V(8KB) + S(8KB) = 32KB
    const int smem_bytes = 4 * TILE_Q * HD * sizeof(__nv_bfloat16);

    cudaFuncSetAttribute(flash_attn_wgmma_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);

    flash_attn_wgmma_kernel<<<grid, block, smem_bytes, stream>>>(
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
    m.def("forward", &attn_forward, "Dense attention BF16 (Round 17: WGMMA m64n64k16 SS-form)");
}
