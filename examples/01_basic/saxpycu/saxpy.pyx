# distutils: language=c++
# distutils: library_dirs=/home/dexter/Projects/setuptools-cuda/examples/saxpycu/saxpycu/
ctypedef fused real:
    double
    float

cdef extern from "saxpy_impl.h":
    void saxpy_wrapper[T](T a, T * x, T * y, int n, int numThreads, int numBlocks);

def saxpy(real a, real[::1] x, real[::1] y, int n, int num_threads, int num_blocks):
    saxpy_wrapper(a, &x[0], &y[0], n, num_threads, num_blocks)
