from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

assert torch.cuda.is_available(), "需要 CUDA 环境来编译"

nvcc_flags = [
    "-O3",
    "-std=c++17",
    "--use_fast_math",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-gencode=arch=compute_80,code=sm_80",
]

cxx_flags = ["-O3", "-std=c++17"]

setup(
    name="attention",
    version="0.1.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[
        CUDAExtension(
            name="attn_cuda",
            sources=["csrc/attention.cu"],
            include_dirs=["csrc"],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
)
