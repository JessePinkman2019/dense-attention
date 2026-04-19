#include "attention.h"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <float.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// ================================================================
// Round 2: Flash Attention with wmma m16n16k16 BF16 Tensor Core
//
// Tile:  TILE_Q=64 (4 warps × m=16), TILE_KV=64, HD=64
// Block: 128 threads (4 warps); warp w handles Q rows [w*16, (w+1)*16)
// Grid:  (ceil(N/TILE_Q), H, B)
// Smem:  Q[64][66] + K[64][66] + V[64][66] + S[64][66] = 33 KB
//        → 3 blocks/SM
//
// wmma fragment layout for accumulator m16n16k16.f32 (8 floats/thread):
//   Thread l holds rows r0=l/4 and r1=l/4+8, for n-tile j:
//   .x[0]=D[r0][2*(l%4)],    .x[1]=D[r0][2*(l%4)+1]     ← left half cols
//   .x[2]=D[r1][2*(l%4)],    .x[3]=D[r1][2*(l%4)+1]
//   .x[4]=D[r0][2*(l%4)+8],  .x[5]=D[r0][2*(l%4)+9]     ← right half cols
//   .x[6]=D[r1][2*(l%4)+8],  .x[7]=D[r1][2*(l%4)+9]
// ================================================================

static constexpr int TILE_Q2  = 64;
static constexpr int TILE_KV2 = 64;
static constexpr int HD2      = 64;
static constexpr int HD_PAD2  = HD2 + 2;    // 66
static constexpr int KV_PAD2  = TILE_KV2 + 2; // 66

__global__ __launch_bounds__(TILE_Q2, 1)
void flash_attn_bf16_mma_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
          __nv_bfloat16* __restrict__ O,
          float*         __restrict__ LSE,
    int N, int H, float scale)
{
    using namespace nvcuda;

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int i_h     = blockIdx.y;
    const int i_b     = blockIdx.z;
    const int q_start = blockIdx.x * TILE_Q2;

    const long stride_H = (long)N * HD2;
    const long stride_B = (long)H * N * HD2;

    const __nv_bfloat16* q_ptr = Q + i_b * stride_B + i_h * stride_H;
    const __nv_bfloat16* k_ptr = K + i_b * stride_B + i_h * stride_H;
    const __nv_bfloat16* v_ptr = V + i_b * stride_B + i_h * stride_H;
          __nv_bfloat16* o_ptr = O + i_b * stride_B + i_h * stride_H;

    // Smem: Q[64][66], K[64][66], V[64][66], S[64][66] = 33 KB
    __shared__ __nv_bfloat16 Q_smem[TILE_Q2 ][HD_PAD2];
    __shared__ __nv_bfloat16 K_smem[TILE_KV2][HD_PAD2];
    __shared__ __nv_bfloat16 V_smem[TILE_KV2][HD_PAD2];
    __shared__ __nv_bfloat16 S_smem[TILE_Q2 ][KV_PAD2];

    // Load Q tile coalesced
    for (int idx = threadIdx.x; idx < TILE_Q2 * HD2; idx += TILE_Q2) {
        const int row = idx / HD2, col = idx % HD2;
        const int g   = q_start + row;
        Q_smem[row][col] = (g < N) ? q_ptr[(long)g * HD2 + col]
                                   : __float2bfloat16(0.f);
    }
    __syncthreads();

    // o_acc[4]: 4 n-tiles of n=16 span HD=64 → output accumulator per warp
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_acc[4];
    for (int j = 0; j < 4; j++) wmma::fill_fragment(o_acc[j], 0.f);

    // Online softmax state: each thread owns rows r0=warp*16+lane/4, r1=r0+8
    float m0 = -FLT_MAX, m1 = -FLT_MAX;
    float l0 = 0.f,      l1 = 0.f;

    const int num_kv = CEIL_DIV(N, TILE_KV2);
    for (int kv = 0; kv < num_kv; ++kv) {
        const int kv_start = kv * TILE_KV2;
        const int kv_len   = min(TILE_KV2, N - kv_start);

        // Load K and V tiles coalesced
        for (int idx = threadIdx.x; idx < TILE_KV2 * HD2; idx += TILE_Q2) {
            const int row = idx / HD2, col = idx % HD2;
            const int g   = kv_start + row;
            const __nv_bfloat16 kv = (row < kv_len) ? k_ptr[(long)g * HD2 + col]
                                                     : __float2bfloat16(0.f);
            K_smem[row][col] = kv;
            V_smem[row][col] = (row < kv_len) ? v_ptr[(long)g * HD2 + col]
                                               : __float2bfloat16(0.f);
        }
        __syncthreads();

        // ---- QK^T via wmma: A=Q[warp*16:+16][k:+16], B=K[n:+16][k:+16] col_major ----
        // col_major B: B[k][n] = K_smem[n_tile*16+n][k_tile*16+k] = K[j][d] → computes QK^T ✓
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> qk_acc[4];
        for (int j = 0; j < 4; j++) wmma::fill_fragment(qk_acc[j], 0.f);

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>    a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major>    bk_frag;
        for (int kt = 0; kt < 4; ++kt) {
            wmma::load_matrix_sync(a_frag, &Q_smem[warp_id * 16][kt * 16], HD_PAD2);
            for (int j = 0; j < 4; ++j) {
                wmma::load_matrix_sync(bk_frag, &K_smem[j * 16][kt * 16], HD_PAD2);
                wmma::mma_sync(qk_acc[j], a_frag, bk_frag, qk_acc[j]);
            }
        }

        // Scale and mask
        for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 8; ++i)
                qk_acc[j].x[i] *= scale;

        if (kv_len < TILE_KV2) {
            for (int j = 0; j < 4; ++j) {
                // x[0],[1]: row r0, cols 2*(l%4), 2*(l%4)+1  in n-tile j
                // x[4],[5]: row r0, cols 2*(l%4)+8, 2*(l%4)+9
                const int c0 = j * 16 + 2 * (lane % 4);
                const int c1 = c0 + 1;
                const int c8 = c0 + 8;
                const int c9 = c1 + 8;
                if (c0 >= kv_len) { qk_acc[j].x[0] = -FLT_MAX; qk_acc[j].x[2] = -FLT_MAX; }
                if (c1 >= kv_len) { qk_acc[j].x[1] = -FLT_MAX; qk_acc[j].x[3] = -FLT_MAX; }
                if (c8 >= kv_len) { qk_acc[j].x[4] = -FLT_MAX; qk_acc[j].x[6] = -FLT_MAX; }
                if (c9 >= kv_len) { qk_acc[j].x[5] = -FLT_MAX; qk_acc[j].x[7] = -FLT_MAX; }
            }
        }

        // Online softmax: row r0 (.x[0,1,4,5]) and r1 (.x[2,3,6,7]) per n-tile j
        float tile_max0 = -FLT_MAX, tile_max1 = -FLT_MAX;
        for (int j = 0; j < 4; ++j) {
            tile_max0 = fmaxf(tile_max0, fmaxf(fmaxf(qk_acc[j].x[0], qk_acc[j].x[1]),
                                                fmaxf(qk_acc[j].x[4], qk_acc[j].x[5])));
            tile_max1 = fmaxf(tile_max1, fmaxf(fmaxf(qk_acc[j].x[2], qk_acc[j].x[3]),
                                                fmaxf(qk_acc[j].x[6], qk_acc[j].x[7])));
        }
        // Reduce within 4-lane row group (lanes l, l^1, l^2, l^3 share same row)
        tile_max0 = fmaxf(tile_max0, __shfl_xor_sync(0xffffffff, tile_max0, 1));
        tile_max0 = fmaxf(tile_max0, __shfl_xor_sync(0xffffffff, tile_max0, 2));
        tile_max1 = fmaxf(tile_max1, __shfl_xor_sync(0xffffffff, tile_max1, 1));
        tile_max1 = fmaxf(tile_max1, __shfl_xor_sync(0xffffffff, tile_max1, 2));

        const float m0_new = fmaxf(m0, tile_max0);
        const float m1_new = fmaxf(m1, tile_max1);
        const float a0 = __expf(m0 - m0_new);
        const float a1 = __expf(m1 - m1_new);

        // Rescale O accumulator and l
        for (int j = 0; j < 4; ++j) {
            o_acc[j].x[0] *= a0; o_acc[j].x[1] *= a0;
            o_acc[j].x[4] *= a0; o_acc[j].x[5] *= a0;
            o_acc[j].x[2] *= a1; o_acc[j].x[3] *= a1;
            o_acc[j].x[6] *= a1; o_acc[j].x[7] *= a1;
        }
        l0 *= a0;
        l1 *= a1;
        m0 = m0_new;
        m1 = m1_new;

        // Compute P = exp(S - m_new), accumulate l, store P to S_smem
        float sum0 = 0.f, sum1 = 0.f;
        const int wr0 = warp_id * 16 + lane / 4;
        const int wr1 = wr0 + 8;
        for (int j = 0; j < 4; ++j) {
            const float p00 = __expf(qk_acc[j].x[0] - m0);
            const float p01 = __expf(qk_acc[j].x[1] - m0);
            const float p04 = __expf(qk_acc[j].x[4] - m0);
            const float p05 = __expf(qk_acc[j].x[5] - m0);
            const float p10 = __expf(qk_acc[j].x[2] - m1);
            const float p11 = __expf(qk_acc[j].x[3] - m1);
            const float p14 = __expf(qk_acc[j].x[6] - m1);
            const float p15 = __expf(qk_acc[j].x[7] - m1);
            sum0 += p00 + p01 + p04 + p05;
            sum1 += p10 + p11 + p14 + p15;

            // Store to S_smem in same accumulator layout
            const int c0 = j * 16 + 2 * (lane % 4);
            S_smem[wr0][c0    ] = __float2bfloat16(p00);
            S_smem[wr0][c0 + 1] = __float2bfloat16(p01);
            S_smem[wr0][c0 + 8] = __float2bfloat16(p04);
            S_smem[wr0][c0 + 9] = __float2bfloat16(p05);
            S_smem[wr1][c0    ] = __float2bfloat16(p10);
            S_smem[wr1][c0 + 1] = __float2bfloat16(p11);
            S_smem[wr1][c0 + 8] = __float2bfloat16(p14);
            S_smem[wr1][c0 + 9] = __float2bfloat16(p15);
        }
        // Reduce l within 4-lane row group
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 1);
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 1);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
        l0 += sum0;
        l1 += sum1;

        // ---- AV via wmma: A=S_smem[warp*16:+16][kt*16:+16], B=V[kt*16:+16][j*16:+16] row_major ----
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> s_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bv_frag;
        for (int kt = 0; kt < 4; ++kt) {
            wmma::load_matrix_sync(s_frag, &S_smem[warp_id * 16][kt * 16], KV_PAD2);
            for (int j = 0; j < 4; ++j) {
                wmma::load_matrix_sync(bv_frag, &V_smem[kt * 16][j * 16], HD_PAD2);
                wmma::mma_sync(o_acc[j], s_frag, bv_frag, o_acc[j]);
            }
        }

        __syncthreads();
    }

    // Normalize and write output
    const float inv_l0 = 1.f / l0;
    const float inv_l1 = 1.f / l1;
    for (int j = 0; j < 4; ++j) {
        o_acc[j].x[0] *= inv_l0; o_acc[j].x[1] *= inv_l0;
        o_acc[j].x[4] *= inv_l0; o_acc[j].x[5] *= inv_l0;
        o_acc[j].x[2] *= inv_l1; o_acc[j].x[3] *= inv_l1;
        o_acc[j].x[6] *= inv_l1; o_acc[j].x[7] *= inv_l1;
    }

    // Each thread writes unique positions — no conflicts
    for (int j = 0; j < 4; ++j) {
        const int row0 = q_start + warp_id * 16 + lane / 4;
        const int row1 = row0 + 8;
        const int c0   = j * 16 + 2 * (lane % 4);
        if (row0 < N) {
            o_ptr[(long)row0 * HD2 + c0    ] = __float2bfloat16(o_acc[j].x[0]);
            o_ptr[(long)row0 * HD2 + c0 + 1] = __float2bfloat16(o_acc[j].x[1]);
            o_ptr[(long)row0 * HD2 + c0 + 8] = __float2bfloat16(o_acc[j].x[4]);
            o_ptr[(long)row0 * HD2 + c0 + 9] = __float2bfloat16(o_acc[j].x[5]);
        }
        if (row1 < N) {
            o_ptr[(long)row1 * HD2 + c0    ] = __float2bfloat16(o_acc[j].x[2]);
            o_ptr[(long)row1 * HD2 + c0 + 1] = __float2bfloat16(o_acc[j].x[3]);
            o_ptr[(long)row1 * HD2 + c0 + 8] = __float2bfloat16(o_acc[j].x[6]);
            o_ptr[(long)row1 * HD2 + c0 + 9] = __float2bfloat16(o_acc[j].x[7]);
        }
    }

    if (LSE && lane % 4 == 0) {
        const int row0 = q_start + warp_id * 16 + lane / 4;
        const int row1 = row0 + 8;
        const long lse_base = (long)i_b * H * N + i_h * N;
        if (row0 < N) LSE[lse_base + row0] = m0 + __logf(l0);
        if (row1 < N) LSE[lse_base + row1] = m1 + __logf(l1);
    }
}

// ================================================================
// Host launchers
// ================================================================

void attention_fwd_bf16(const AttentionParams& p, cudaStream_t stream)
{
    TORCH_CHECK(p.head_dim == HD2,
                "head_dim must be ", HD2, " for this kernel, got ", p.head_dim);
    const dim3 grid(CEIL_DIV(p.seq_len, TILE_Q2), p.num_heads, p.batch_size);
    flash_attn_bf16_mma_kernel<<<grid, dim3(TILE_Q2), 0, stream>>>(
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
    m.def("forward", &attn_forward, "Dense attention forward BF16 (Round 2 wmma m16n16k16)");
}
