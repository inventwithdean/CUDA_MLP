#include "matmul.h"

__global__ void matmul_kernel(float *a, float *b, float *c, int M, int N, int P)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    float tmp_sum = 0;
    if (row < M && col < P)
    {
        for (int k = 0; k < N; k++)
        {
            tmp_sum += a[(row * N + k)] * b[(k * P + col)];
        }
        c[row * P + col] = tmp_sum;
    }
}

void matmul(float *a, float *b, float *c, int M, int N, int P)
{
    // OUT SHAPE: (M x P)
    // x idx should at least be P
    // y idx should at least be M
    dim3 threadSize(16, 16);

    dim3 numBlocks(P > 16 ? ceil(float(P) / 16) : 1, M > 16 ? ceil(float(M) / 16) : 1);
    matmul_kernel<<<numBlocks, threadSize>>>(a, b, c, M, N, P);
    cudaDeviceSynchronize();
}