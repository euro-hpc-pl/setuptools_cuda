"""Setup module for the `example` package."""
from setuptools import setup

from setuptools_cuda import CudaExtension

setup(
    cuda_extensions=[
        CudaExtension(
            name="saxpycu",
            sources=["saxpycu/saxpy.pyx", "saxpycu/saxpy_impl.cu"],
        ),
    ],
)
