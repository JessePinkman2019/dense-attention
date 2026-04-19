#include "attention.h"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <float.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// ================================================================
// Round 1: scalar FP32 Flash Attention baseline
// TILE_Q=128 query rows/block, TILE_KV=64, HD=64
// Block: 128 threads; 1 thread per Q row; online softmax (1-pass)
// Smem: Q[128][66] + K[64][66] + V[64][66] = 33 KB/block → 3 blocks/SM
// HD_PAD=66: row stride = 132B = 33 banks (odd) → no bank conflicts
// ================================================================

static constexpr int TILE_Q  = 128;
static constexpr int TILE_KV = 64;
static constexpr int HD      = 64;
static constexpr int HD_PAD  = HD + 2;   // 66

__global__ __launch_bounds__(TILE_Q, 1)
void flash_attn_bf16_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
          __nv_bfloat16* __restrict__ O,
          float*         __restrict__ LSE,
    int N, int H, float scale)
{
    const int tid = threadIdx.x;
    const int i_q = blockIdx.x * TILE_Q + tid;
    const int i_h = blockIdx.y;
    const int i_b = blockIdx.z;

    const long stride_H = (long)N * HD;
    const long stride_B = (long)H * N * HD;

    const __nv_bfloat16* q_ptr = Q + i_b * stride_B + i_h * stride_H;
    const __nv_bfloat16* k_ptr = K + i_b * stride_B + i_h * stride_H;
    const __nv_bfloat16* v_ptr = V + i_b * stride_B + i_h * stride_H;
          __nv_bfloat16* o_ptr = O + i_b * stride_B + i_h * stride_H;

    __shared__ __nv_bfloat16 Q_smem[TILE_Q ][HD_PAD];
    __shared__ __nv_bfloat16 K_smem[TILE_KV][HD_PAD];
    __shared__ __nv_bfloat16 V_smem[TILE_KV][HD_PAD];

    // Load Q tile coalesced
    for (int idx = tid; idx < TILE_Q * HD; idx += TILE_Q) {
        const int row = idx / HD, col = idx % HD;
        const int g   = blockIdx.x * TILE_Q + row;
        Q_smem[row][col] = (g < N) ? q_ptr[(long)g * HD + col]
                                   : __float2bfloat16(0.f);
    }
    __syncthreads();

    float o_reg[HD];
    #pragma unroll
    for (int d = 0; d < HD; d++) o_reg[d] = 0.f;
    float m_val = -FLT_MAX;
    float l_val = 0.f;

    const int num_kv = CEIL_DIV(N, TILE_KV);
    for (int kv = 0; kv < num_kv; ++kv) {
        const int kv_start = kv * TILE_KV;
        const int kv_len   = min(TILE_KV, N - kv_start);

        for (int idx = tid; idx < TILE_KV * HD; idx += TILE_Q) {
            const int row = idx / HD, col = idx % HD;
            const int g   = kv_start + row;
            K_smem[row][col] = (row < kv_len) ? k_ptr[(long)g * HD + col]
                                               : __float2bfloat16(0.f);
            V_smem[row][col] = (row < kv_len) ? v_ptr[(long)g * HD + col]
                                               : __float2bfloat16(0.f);
        }
        __syncthreads();

        if (i_q < N) {
            for (int j = 0; j < kv_len; ++j) {
                float s = 0.f;
                #pragma unroll
                for (int d = 0; d < HD; ++d)
                    s += (float)Q_smem[tid][d] * (float)K_smem[j][d];
                s *= scale;

                const float m_new = fmaxf(m_val, s);
                const float alpha  = __expf(m_val - m_new);
                const float p      = __expf(s - m_new);

                #pragma unroll
                for (int d = 0; d < HD; ++d)
                    o_reg[d] = o_reg[d] * alpha + p * (float)V_smem[j][d];

                l_val = l_val * alpha + p;
                m_val = m_new;
            }
        }
        __syncthreads();
    }

    if (i_q < N) {
        const float inv_l = 1.f / l_val;
        #pragma unroll
        for (int d = 0; d < HD; ++d)
            o_ptr[(long)i_q * HD + d] = __float2bfloat16(o_reg[d] * inv_l);

        if (LSE)
            LSE[(long)i_b * H * N + i_h * N + i_q] = m_val + __logf(l_val);
    }
}

void attention_fwd_bf16(const AttentionParams& p, cudaStream_t stream)
{
    TORCH_CHECK(p.head_dim == HD,
                "head_dim must be ", HD, " for Round-1 kernel, got ", p.head_dim);
    const dim3 grid(CEIL_DIV(p.seq_len, TILE_Q), p.num_heads, p.batch_size);
    flash_attn_bf16_kernel<<<grid, dim3(TILE_Q), 0, stream>>>(
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
    m.def("forward", &attn_forward, "Dense attention forward BF16 (Round 1 scalar baseline)");
}
