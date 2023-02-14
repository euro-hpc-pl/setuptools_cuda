"""Setup module for the `example` package."""
from setuptools import setup

from setuptools_cuda import CudaExtension

setup(
    cuda_extensions=[
        CudaExtension(
            name="saxpycu_ext",
            sources=["saxpycu/ext/saxpy.pyx", "saxpycu/ext/saxpy_impl.cu"],
        ),
    ],
)
