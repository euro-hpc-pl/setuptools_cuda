# Setuptools plugin for CUDA extensions

The `setuptools-cuda` is a `setuptools` plugin for building CUDA enabled Python extension modules.

## How does it compare to other packages on the market?

As far as the authors of this package know, other CUDA-oriented Python projects focus mostly on providing 
higher-level abstractions over CUDA that can be accessed in Python. For instance, the well-known
[PyCUDA](https://pypi.org/project/pycuda/) provides `GPUArray` and `SourceModule` abstractions.

However, when it comes to compiling extension modules that use CUDA, surprisingly there seems to be no good
solution that just works out of the box. Typically, people tend to integrate the CUDA code into their extension
modules either using some third-party build systems or by writing some ad-hoc hacks for setuptools (see e.g.
[this](https://stackoverflow.com/questions/10034325/can-python-distutils-compile-cuda-code) StakOverflow question.

The `setuptools-cuda` tries to fill this niche. It allows one for defining extension modules containing `.cu`
compilation units that will be compiled with `nvcc`. Such extensions can then be build using normal `setuptools`
build procedures.

## Quickstart

Using `setuptools-cuda` is easy and requires you to perform the following steps.

1. Add `setuptools-cuda` to your `build-system` requirements in `pyproject.toml`. For instance like this:

   ```toml
   [build-system]
   requires = ["setuptools", "wheel", "cython", "setuptools-cuda"]
   ```

   If you are not using isolated builds, you should install `setuptools-cuda` in
   your environment using `pip`.

2. Declare your extension module by passing list of `CUDAExtension` objects to `cuda_extensions` keyword to
   the `setup()` function call in `setup.py`. For instance, one of the examples in this repository has
   the following `setup.py` file:

   ```python
   from setuptools import setup

   from setuptools_cuda import CudaExtension

   setup(
       cuda_extensions=[
           CudaExtension(
               name="thrust",
               sources=["thrustcu/thrustcu.pyx", "thrustcu/thrustcu_impl.cu"],
           ),
       ],
   )
   ```
3. **IMPORTANT** define `CUDAHOME` environment variable. It should point to the CUDA installation location. E.g.
   ```bash
   export CUDAHOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda
   ```
   If you won't define the `CUDAHOME` evironmental variable, `setuptools-cuda` will do its best to guess it, but
   our experience shows that it might fail miserably (and probably silently).
4. Build your package as usual. Typically just running `pip install` should do.

## Acknowledgements

This package was inspired by [setuptools-rust](https://github.com/PyO3/setuptools-rust) package.
