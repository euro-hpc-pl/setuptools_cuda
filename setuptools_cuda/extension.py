"""Definition of CUDA-enabled extensions."""
from setuptools import Extension

DEFAULT_CUDA_LIBS = ["cudart"]


class CudaExtension(Extension):
    """CUDA-enabled extension."""

    def __init__(self, name, sources, extra_nvcc_args=None, *args, **kwargs):
        super().__init__(name, sources, *args, **kwargs)
        self.libraries += DEFAULT_CUDA_LIBS
        self.extra_nvcc_args = [] if extra_nvcc_args is None else extra_nvcc_args
