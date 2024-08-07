#include "mat_axis_sum.h"

__global__ void mat_axis_sum_kernel(float *mat, float *out, int M, int N)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col < N)
    {
        float sum = 0;
        for (int row = 0; row < M; row++)
        {
            sum += mat[row * N + col];
        }
        out[col] = sum;
    }
}

// Supports only Axis 0 currently.
Matrix *mat_axis_sum(Matrix *mat, int axis)
{
    int M = mat->rows;
    int N = mat->cols;
    Matrix *out = new Matrix(1, N);
    dim3 threadSize(256);
    dim3 blockSize(N > 256 ? ceil(N / 256) : 1);
    mat_axis_sum_kernel<<<blockSize, threadSize>>>(mat->mat, out->mat, M, N);
    cudaDeviceSynchronize();
    return out;
}