# distutils: language=c++
cimport numpy as np

ctypedef fused number:
    double
    float
    np.int8_t
    np.uint8_t
    np.int16_t
    np.uint16_t
    np.int32_t
    np.uint32_t
    np.int64_t
    np.uint64_t


cdef extern from "thrustcu_impl.h":
    void _sort[T](T* data, long long n);

def sort(number[::1] data):
    _sort(&data[0], len(data))
