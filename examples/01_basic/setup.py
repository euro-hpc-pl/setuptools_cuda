"""Setup module for the `example` package."""
from Cython.Build import cythonize
from setuptools import setup

from setuptools_cuda import CudaExtension

setup(
    cuda_extensions=cythonize(
        [
            CudaExtension(
                name="saxpycu",
                sources=["saxpycu/saxpy.pyx", "saxpycu/saxpy_impl.cu"],
            ),
        ]
    ),
)
