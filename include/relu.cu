#include "relu.h"

ReLU::ReLU(int M, int N) : M(M), N(N)
{
    size_t input_size = M * N * sizeof(float); // Same as output_size
    cudaMallocManaged(&output, input_size);
    cudaMallocManaged(&input, input_size);
}

__global__ void ReLU_kernel(float *input, float *output, int M, int N)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = row * N + col;
    if (idx < M * N)
    {
        output[idx] = input[idx] > 0 ? input[idx] : 0;
    }
}

__global__ void ReLU_backwards_kernel(float *input, float *grads, int M, int N)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = row * N + col;
    if (idx < M * N)
    {
        grads[idx] = input[idx] >= 0 ? grads[idx] : 0;
    }
}

float *ReLU::forward(float *input)
{
    cudaMemcpy(this->input, input, M * N * sizeof(float), cudaMemcpyHostToDevice);
    dim3 threadSize(16, 16);
    dim3 blockSize(N > 16 ? ceil(float(N)) : 1, M > 16 ? ceil(float(M)) : 1);
    ReLU_kernel<<<blockSize, threadSize>>>(input, output, M, N);
    cudaDeviceSynchronize();
    return output;
}

void ReLU::backward(float *grads)
{
    dim3 threadSize(16, 16);
    dim3 blockSize(N > 16 ? ceil(float(N)) : 1, M > 16 ? ceil(float(M)) : 1);
    ReLU_backwards_kernel<<<blockSize, threadSize>>>(input, grads, M, N);
    ;
}

ReLU::~ReLU()
{
    cudaFree(input);
    cudaFree(output);
}
