#include "mat_add.h"

__global__ void mat_add_broadcasted_kernel(float *weights, float *biases, float *out, int M, int N)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < M && col < N)
    {
        int idx = row * N + col;
        out[idx] = weights[idx] + biases[col];
    }
}

void mat_add_broadcasted(float *weights, float *biases, float *out, int M, int N)
// Adds weights to biases by broadcasting biases from 1 by N to M by N
{
    dim3 threadSize(16, 16);
    dim3 numBlocks(N > 16 ? ceil(float(N) / 16) : 1, M > 16 ? ceil(float(M) / 16) : 1);
    mat_add_broadcasted_kernel<<<threadSize, numBlocks>>>(weights, biases, out, M, N);
    cudaDeviceSynchronize();
}