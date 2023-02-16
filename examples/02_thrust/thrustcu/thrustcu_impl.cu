#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>


template <typename T>
void _sort(T* data, long long n)
{
  thrust::device_vector<T> data_vec(data, data+n);
  thrust::sort(data_vec.begin(), data_vec.end());
  thrust::copy(data_vec.begin(), data_vec.end(), data);
}

template void _sort(float*, long long);
template void _sort(double*, long long);
template void _sort(int8_t*, long long);
template void _sort(uint8_t*, long long);
template void _sort(int16_t*, long long);
template void _sort(uint16_t*, long long);
template void _sort(int32_t*, long long);
template void _sort(uint32_t*, long long);
template void _sort(int64_t*, long long);
template void _sort(uint64_t*, long long);
