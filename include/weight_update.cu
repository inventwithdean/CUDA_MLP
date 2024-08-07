#include "weight_update.h"

__global__ void update_parameters_kernel(float *weights, float *grads, int M, int N, float lr)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = row * N + col;
    if (row < M & col < N)
    {
        weights[idx] -= lr * grads[idx];
    }
}

void update_parameters(Matrix *weight_matrix, Matrix *grad_matrix, float learning_rate)
{
    int M = weight_matrix->rows;
    int N = weight_matrix->cols;
    dim3 threadSize(16, 16);
    dim3 numBlocks(N > 16 ? ceil(float(N) / 16) : 1, M > 16 ? ceil(float(M) / 16) : 1);

    update_parameters_kernel<<<threadSize, numBlocks>>>(weight_matrix->mat, grad_matrix->mat, M, N, learning_rate);
    cudaDeviceSynchronize();
}