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

Matrix *mat_transpose(Matrix *mat)
{
    int rows = mat->rows;
    int cols = mat->cols;
    Matrix *transposed_mat = new Matrix(cols, rows);
    dim3 numThreads(16, 16);
    dim3 numBlocks(cols > 16 ? ceil(float(cols) / 16) : 1, rows > 16 ? ceil(float(rows) / 16) : 1);
    mat_transpose_kernel<<<numThreads, numBlocks>>>(mat->mat, transposed_mat->mat, rows, cols);
    cudaDeviceSynchronize();
    return transposed_mat;
}
