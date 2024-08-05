#include "relu.h"

ReLU::ReLU(int M, int N) : M(M), N(N)
{
    size_t out_size = M * N * sizeof(float);
    cudaMallocManaged(&(this->output), out_size);
}

__global__ void ReLU_kernel(float *input, float *output, int M, int N)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = row * N + col;
    output[idx] = input[idx] > 0 ? input[idx] : 0;
}

float *ReLU::forward(float *input)
{
    dim3 threadSize(16, 16);
    dim3 blockSize(N > 16 ? ceil(float(N)) : 1, M > 16 ? ceil(float(M)) : 1);
    ReLU_kernel<<<blockSize, threadSize>>>(input, output, M, N);
    cudaDeviceSynchronize();
    return output;
}

ReLU::~ReLU()
{
    cudaFree(output);
}
