"""Setup module for the `example` package."""
import numpy as np
from setuptools import setup

from setuptools_cuda import CudaExtension

setup(
    cuda_extensions=[
        CudaExtension(
            name="thrust",
            sources=["thrustcu/thrustcu.pyx", "thrustcu/thrustcu_impl.cu"],
            include_dirs=[np.get_include()],
        ),
    ],
)
