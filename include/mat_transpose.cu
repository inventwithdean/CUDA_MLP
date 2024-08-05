#include "mat_transpose.h"

__global__ void mat_transpose_kernel(float *matrix, float *transposed_matrix, int M, int N)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (row < M && col < N)
    {
        transposed_matrix[col * M + row] = matrix[row * N + col];
    }
}

float *mat_transpose(float *matrix, int M, int N)
{
    size_t size_mat = M * N * sizeof(float);
    float *transposed_mat;
    cudaError_t err;
    err = cudaMallocManaged(&transposed_mat, size_mat);
    if (err != cudaSuccess)
    {
        printf("cudaMallocManaged Failed: %s\n", cudaGetErrorString(err));
    }
    dim3 numThreads(16, 16);
    dim3 numBlocks(N > 16 ? ceil(float(N) / 16) : 1, M > 16 ? ceil(float(M) / 16) : 1);
    mat_transpose_kernel<<<numThreads, numBlocks>>>(matrix, transposed_mat, M, N);
    cudaDeviceSynchronize();
    return transposed_mat;
}
