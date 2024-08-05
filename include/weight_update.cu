#include "weight_update.h"

__global__ void update_weights_kernel(float *weights, float *grads, int M, int N, float lr)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = row * N + col;
    if (row < M & col < N)
    {
        weights[idx] -= lr * grads[idx];
    }
}

void update_weights(float *weights, float *grads, int M, int N, float learning_rate)
{
    dim3 threadSize(16, 16);
    dim3 numBlocks(N > 16 ? ceil(float(N) / 16) : 1, M > 16 ? ceil(float(M) / 16) : 1);

    update_weights_kernel<<<threadSize, numBlocks>>>(weights, grads, M, N, learning_rate);
    cudaDeviceSynchronize();
}