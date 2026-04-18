#include "attention.h"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <float.h>

void attention_fwd_bf16(
    const AttentionParams& params,
    cudaStream_t stream
) {
    // TODO
}

void attention_fwd_fp16(
    const AttentionParams& params,
    cudaStream_t stream
) {
    // TODO
}
