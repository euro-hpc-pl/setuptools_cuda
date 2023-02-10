import os
import re
import subprocess
from distutils.spawn import find_executable


def find_cuda_home():
    if "CUDAHOME" in os.environ:
        cuda_home = os.environ["CUDAHOME"]
    else:
        nvcc_path = find_executable("nvcc")
        if not nvcc_path:
            raise ValueError("It appears that you don't have nvcc in your PATH.")
        cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
    return cuda_home


def get_cuda_version():
    nvcc_proc = subprocess.Popen(["nvcc", "--version"], stdout=subprocess.PIPE)
    nvcc_proc.wait()
    nvcc_version_string = nvcc_proc.stdout.read().decode()
    return next(iter(re.search(r"release (\d+\.\d+)", nvcc_version_string).groups()))
