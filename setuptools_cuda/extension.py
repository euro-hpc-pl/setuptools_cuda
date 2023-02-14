"""Definition of CUDA-enabled extensions."""
from setuptools import Extension


class CudaExtension(Extension):
    """CUDA-enabled extension."""

    def __init__(self, name, sources, extra_nvcc_args=None, *args, **kwargs):
        super().__init__(name, sources, *args, **kwargs)
        self.extra_nvcc_args = [] if extra_nvcc_args is None else extra_nvcc_args
