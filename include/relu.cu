#include "relu.h"

ReLU::ReLU()
{
    input = nullptr;
    output = nullptr;
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

Matrix *ReLU::forward(Matrix *input)
{

    this->input = input;
    int M = input->rows;
    int N = input->cols;
    if (output == nullptr)
    {
        output = new Matrix(M, N);
    }
    dim3 threadSize(16, 16);
    dim3 blockSize(N > 16 ? ceil(float(N)) : 1, M > 16 ? ceil(float(M)) : 1);
    ReLU_kernel<<<blockSize, threadSize>>>(input->mat, output->mat, M, N);
    cudaDeviceSynchronize();
    return output;
}

void ReLU::backward(Matrix *grads)
{
    int M = grads->rows;
    int N = grads->cols;
    dim3 threadSize(16, 16);
    dim3 blockSize(N > 16 ? ceil(float(N)) : 1, M > 16 ? ceil(float(M)) : 1);
    ReLU_backwards_kernel<<<blockSize, threadSize>>>(input->mat, grads->mat, M, N);
    cudaDeviceSynchronize();
}

ReLU::~ReLU()
{
    delete (input);
    delete (output);
}
