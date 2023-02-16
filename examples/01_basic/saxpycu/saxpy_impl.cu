template <typename T>
__global__ void _saxpy(T a, T* x, T* y, int n)
{
  int stride = gridDim.x * blockDim.x;
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = id; i < n; i += stride)
  {
    y[i] = a * x[i] + y[i];
  }
}

template <typename T>
void saxpy_wrapper(T a, T* x, T* y, int n, int numThreads, int numBlocks)
{
  T *dx, *dy;

  cudaMalloc(&dx, n * sizeof(T));
  cudaMalloc(&dy, n * sizeof(T));

  cudaMemcpy(dx, x, n * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(dy, y, n * sizeof(T), cudaMemcpyHostToDevice);

  _saxpy<<<numThreads, numBlocks>>>(a, dx, dy, n);

  cudaMemcpy(y, dy, n * sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(dx);
  cudaFree(dy);
}

template void saxpy_wrapper(float, float*, float*, int, int, int);
template void saxpy_wrapper(double, double*, double*, int, int, int);
