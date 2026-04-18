#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

struct AttentionParams {
    const void* q;    // [B, H, N, D]
    const void* k;    // [B, H, N, D]
    const void* v;    // [B, H, N, D]

    void* out;        // [B, H, N, D]
    float* lse;       // [B, H, N]

    int batch_size;
    int num_heads;
    int seq_len;
    int head_dim;

    float scale;
    bool is_training;
};

void attention_fwd_bf16(
    const AttentionParams& params,
    cudaStream_t stream
);

void attention_fwd_fp16(
    const AttentionParams& params,
    cudaStream_t stream
);
