#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <float.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
                   cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// 数值类型转换

__device__ __forceinline__ float to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
__device__ __forceinline__ float to_float(__half x) {
    return __half2float(x);
}
__device__ __forceinline__ float to_float(float x) {
    return x;
}

__device__ __forceinline__ __nv_bfloat16 from_float_bf16(float x) {
    return __float2bfloat16(x);
}
__device__ __forceinline__ __half from_float_fp16(float x) {
    return __float2half(x);
}
