import os
from distutils.command.build_ext import build_ext
from typing import List, Literal, Type, cast

from Cython.Build import cythonize
from setuptools.dist import Distribution

from .compiler_customization import customize_compiler_for_nvcc
from .extension import CudaExtension
from .inspections import find_cuda_home


def add_cuda_extensions(dist: Distribution) -> None:
    build_ext_base_class = cast(Type[build_ext], dist.cmdclass.get("build_ext", build_ext))

    class BuildCudaExtension(build_ext_base_class):  # type: ignore[valid-type, misc]
        def run(self):
            super().run()

        def build_extensions(self):
            customize_compiler_for_nvcc(self.compiler)

            try:
                cuda_home = find_cuda_home()
                self.compiler.has_nvcc = True
                for ext in self.extensions:
                    ext.include_dirs += [os.path.join(cuda_home, "include")]
                    ext.library_dirs += [
                        os.path.join(cuda_home, "lib64"),
                        os.path.join(cuda_home, "lib"),
                    ]
            except ValueError:
                self.compiler.has_nvcc = False

            build_ext.build_extensions(self)

    dist.cmdclass["build_ext"] = BuildCudaExtension
    cythonized_cuda_extensions = cythonize(dist.cuda_extensions)
    dist.ext_modules = (
        cythonized_cuda_extensions
        if dist.ext_modules is None
        else dist.ext_modules + cythonized_cuda_extensions
    )
    print(dist.ext_modules)


def cuda_extensions(
    dist: Distribution, attr: Literal["cuda_extensions"], value: List[CudaExtension]
) -> None:
    assert attr == "cuda_extensions"
    has_cuda_extensions = len(value) > 0

    old_has_ext_modules = dist.has_ext_modules

    def has_ext_modules():
        return old_has_ext_modules() or has_cuda_extensions

    dist.has_ext_modules = has_ext_modules
    add_cuda_extensions(dist)
