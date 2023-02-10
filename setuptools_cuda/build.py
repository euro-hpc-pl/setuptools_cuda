from setuptools.command.build_ext import build_ext


class build_cuda_ext(build_ext):
    def __init__(self, dist):
        super().__init__(dist)

    def build_extension(self, ext) -> None:
        super().build_extension(ext)
