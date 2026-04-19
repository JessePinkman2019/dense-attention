#include "attention.h"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <float.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// ================================================================
// Round 6: wmma m16n16k16 BF16, TILE_KV=16
//
// Further register reduction: TILE_KV=16 → qk_acc[1] (single tile)
// Smem: Q[64][66]+K[16][66]+V[16][66]+S[64][18] ≈ 19KB
// Expected: ~64-80 registers/thread → 35-43% occupancy
// ================================================================

static constexpr int TILE_Q6  = 64;
static constexpr int TILE_KV6 = 16;   // reduced from 32
static constexpr int HD6      = 64;
static constexpr int HD_PAD6  = HD6 + 2;        // 66
static constexpr int KV_PAD6  = TILE_KV6 + 2;   // 18
static constexpr int BLK6     = 128;

#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#endif

__global__ void flash_attn_bf16_mma6_kernel(
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
    const int q_start = blockIdx.x * TILE_Q6;

    const long stride_H = (long)N  * HD6;
    const long stride_B = (long)H * N * HD6;

    const __nv_bfloat16* q_ptr = Q + i_b * stride_B + i_h * stride_H;
    const __nv_bfloat16* k_ptr = K + i_b * stride_B + i_h * stride_H;
    const __nv_bfloat16* v_ptr = V + i_b * stride_B + i_h * stride_H;
          __nv_bfloat16* o_ptr = O + i_b * stride_B + i_h * stride_H;

    // Smem ≈ 19KB
    __shared__ __nv_bfloat16 Q_smem[TILE_Q6 ][HD_PAD6];   // [64][66]
    __shared__ __nv_bfloat16 K_smem[TILE_KV6][HD_PAD6];   // [16][66]
    __shared__ __nv_bfloat16 V_smem[TILE_KV6][HD_PAD6];   // [16][66]
    __shared__ __nv_bfloat16 S_smem[TILE_Q6 ][KV_PAD6];   // [64][18]

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_acc[4];
    for (int j = 0; j < 4; j++) wmma::fill_fragment(o_acc[j], 0.f);

    float m0 = -FLT_MAX, m1 = -FLT_MAX;
    float l0 = 0.f,      l1 = 0.f;

    for (int idx = threadIdx.x; idx < TILE_Q6 * HD6; idx += BLK6) {
        const int r = idx / HD6, c = idx % HD6;
        const int g = q_start + r;
        Q_smem[r][c] = (g < N) ? q_ptr[(long)g * HD6 + c] : __float2bfloat16(0.f);
    }
    __syncthreads();

    const int num_kv = CEIL_DIV(N, TILE_KV6);
    for (int kv = 0; kv < num_kv; ++kv) {
        const int kv_start = kv * TILE_KV6;
        const int kv_len   = min(TILE_KV6, N - kv_start);

        for (int idx = threadIdx.x; idx < TILE_KV6 * HD6; idx += BLK6) {
            const int r = idx / HD6, c = idx % HD6;
            const int g = kv_start + r;
            K_smem[r][c] = (r < kv_len) ? k_ptr[(long)g * HD6 + c] : __float2bfloat16(0.f);
            V_smem[r][c] = (r < kv_len) ? v_ptr[(long)g * HD6 + c] : __float2bfloat16(0.f);
        }
        __syncthreads();

        // ---- QK^T: TILE_KV6/16 = 1 n-tile (qk_acc[1]) ----
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> qk_acc[1];
        wmma::fill_fragment(qk_acc[0], 0.f);

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>    a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major>    bk_frag;
        for (int kt = 0; kt < 4; ++kt) {
            wmma::load_matrix_sync(a_frag, &Q_smem[warp_id * 16][kt * 16], HD_PAD6);
            wmma::load_matrix_sync(bk_frag, &K_smem[0][kt * 16], HD_PAD6);  // single n-tile
            wmma::mma_sync(qk_acc[0], a_frag, bk_frag, qk_acc[0]);
        }

        for (int i = 0; i < 8; ++i)
            qk_acc[0].x[i] *= scale;

        if (kv_len < TILE_KV6) {
            const int c0 = 2 * (lane % 4);
            if (c0   >= kv_len) { qk_acc[0].x[0]=-FLT_MAX; qk_acc[0].x[2]=-FLT_MAX; }
            if (c0+1 >= kv_len) { qk_acc[0].x[1]=-FLT_MAX; qk_acc[0].x[3]=-FLT_MAX; }
            if (c0+8 >= kv_len) { qk_acc[0].x[4]=-FLT_MAX; qk_acc[0].x[6]=-FLT_MAX; }
            if (c0+9 >= kv_len) { qk_acc[0].x[5]=-FLT_MAX; qk_acc[0].x[7]=-FLT_MAX; }
        }

        float tile_max0 = fmaxf(fmaxf(qk_acc[0].x[0], qk_acc[0].x[1]),
                                fmaxf(qk_acc[0].x[4], qk_acc[0].x[5]));
        float tile_max1 = fmaxf(fmaxf(qk_acc[0].x[2], qk_acc[0].x[3]),
                                fmaxf(qk_acc[0].x[6], qk_acc[0].x[7]));
        tile_max0 = fmaxf(tile_max0, __shfl_xor_sync(0xffffffff, tile_max0, 1));
        tile_max0 = fmaxf(tile_max0, __shfl_xor_sync(0xffffffff, tile_max0, 2));
        tile_max1 = fmaxf(tile_max1, __shfl_xor_sync(0xffffffff, tile_max1, 1));
        tile_max1 = fmaxf(tile_max1, __shfl_xor_sync(0xffffffff, tile_max1, 2));

        const float m0_new = fmaxf(m0, tile_max0);
        const float m1_new = fmaxf(m1, tile_max1);
        const float a0_s = __expf(m0 - m0_new);
        const float a1_s = __expf(m1 - m1_new);

        for (int j = 0; j < 4; ++j) {
            o_acc[j].x[0] *= a0_s; o_acc[j].x[1] *= a0_s;
            o_acc[j].x[4] *= a0_s; o_acc[j].x[5] *= a0_s;
            o_acc[j].x[2] *= a1_s; o_acc[j].x[3] *= a1_s;
            o_acc[j].x[6] *= a1_s; o_acc[j].x[7] *= a1_s;
        }
        l0 *= a0_s; m0 = m0_new;
        l1 *= a1_s; m1 = m1_new;

        float sum0 = 0.f, sum1 = 0.f;
        const int wr0 = warp_id * 16 + lane / 4;
        const int wr1 = wr0 + 8;
        const int c0 = 2 * (lane % 4);  // single n-tile (j=0)
        const float p0 = __expf(qk_acc[0].x[0] - m0);
        const float p1 = __expf(qk_acc[0].x[1] - m0);
        const float p4 = __expf(qk_acc[0].x[4] - m0);
        const float p5 = __expf(qk_acc[0].x[5] - m0);
        sum0 += p0 + p1 + p4 + p5;
        S_smem[wr0][c0    ] = __float2bfloat16(p0);
        S_smem[wr0][c0 + 1] = __float2bfloat16(p1);
        S_smem[wr0][c0 + 8] = __float2bfloat16(p4);
        S_smem[wr0][c0 + 9] = __float2bfloat16(p5);
        const float p2 = __expf(qk_acc[0].x[2] - m1);
        const float p3 = __expf(qk_acc[0].x[3] - m1);
        const float p6 = __expf(qk_acc[0].x[6] - m1);
        const float p7 = __expf(qk_acc[0].x[7] - m1);
        sum1 += p2 + p3 + p6 + p7;
        S_smem[wr1][c0    ] = __float2bfloat16(p2);
        S_smem[wr1][c0 + 1] = __float2bfloat16(p3);
        S_smem[wr1][c0 + 8] = __float2bfloat16(p6);
        S_smem[wr1][c0 + 9] = __float2bfloat16(p7);
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 1);
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 1);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
        l0 += sum0; l1 += sum1;

        // ---- AV: TILE_KV6/16 = 1 k-tile ----
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> s_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bv_frag;
        wmma::load_matrix_sync(s_frag, &S_smem[warp_id * 16][0], KV_PAD6);  // single k-tile
        for (int j = 0; j < 4; ++j) {
            wmma::load_matrix_sync(bv_frag, &V_smem[0][j * 16], HD_PAD6);
            wmma::mma_sync(o_acc[j], s_frag, bv_frag, o_acc[j]);
        }

        __syncthreads();
    }

    const float inv_l0 = 1.f / l0;
    const float inv_l1 = 1.f / l1;
    for (int j = 0; j < 4; ++j) {
        const int row0 = q_start + warp_id * 16 + lane / 4;
        const int row1 = row0 + 8;
        const int cb   = j * 16 + 2 * (lane % 4);
        if (row0 < N) {
            o_ptr[(long)row0 * HD6 + cb    ] = __float2bfloat16(o_acc[j].x[0] * inv_l0);
            o_ptr[(long)row0 * HD6 + cb + 1] = __float2bfloat16(o_acc[j].x[1] * inv_l0);
            o_ptr[(long)row0 * HD6 + cb + 8] = __float2bfloat16(o_acc[j].x[4] * inv_l0);
            o_ptr[(long)row0 * HD6 + cb + 9] = __float2bfloat16(o_acc[j].x[5] * inv_l0);
        }
        if (row1 < N) {
            o_ptr[(long)row1 * HD6 + cb    ] = __float2bfloat16(o_acc[j].x[2] * inv_l1);
            o_ptr[(long)row1 * HD6 + cb + 1] = __float2bfloat16(o_acc[j].x[3] * inv_l1);
            o_ptr[(long)row1 * HD6 + cb + 8] = __float2bfloat16(o_acc[j].x[6] * inv_l1);
            o_ptr[(long)row1 * HD6 + cb + 9] = __float2bfloat16(o_acc[j].x[7] * inv_l1);
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

void attention_fwd_bf16(const AttentionParams& p, cudaStream_t stream)
{
    TORCH_CHECK(p.head_dim == HD6,
                "head_dim must be ", HD6, " for this kernel, got ", p.head_dim);
    const dim3 grid(CEIL_DIV(p.seq_len, TILE_Q6), p.num_heads, p.batch_size);
    flash_attn_bf16_mma6_kernel<<<grid, dim3(BLK6), 0, stream>>>(
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
    m.def("forward", &attn_forward, "Dense attention BF16 (Round 6: TILE_KV=16)");
}
