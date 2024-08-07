#include "matrix_calc.h"

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

__global__ void mat_transpose_kernel(float *matrix, float *transposed_matrix, int M, int N)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (row < M && col < N)
    {
        transposed_matrix[col * M + row] = matrix[row * N + col];
    }
}

void transpose_matrix(Matrix *mat, Matrix *out)
{
    int rows = mat->rows;
    int cols = mat->cols;
    dim3 numThreads(16, 16);
    dim3 numBlocks(cols > 16 ? ceil(float(cols) / 16) : 1, rows > 16 ? ceil(float(rows) / 16) : 1);
    mat_transpose_kernel<<<numThreads, numBlocks>>>(mat->mat, out->mat, rows, cols);
    cudaDeviceSynchronize();
}

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

// UPDATE THIS FUNCTION!
void mat_add_broadcasted(float *weights, float *biases, float *out, int M, int N)
// Adds weights to biases by broadcasting biases from 1 by N to M by N
{
    dim3 threadSize(16, 16);
    dim3 numBlocks(N > 16 ? ceil(float(N) / 16) : 1, M > 16 ? ceil(float(M) / 16) : 1);
    mat_add_broadcasted_kernel<<<threadSize, numBlocks>>>(weights, biases, out, M, N);
    cudaDeviceSynchronize();
}

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
void mat_axis_sum(Matrix *mat, Matrix *out, int axis)
{
    int M = mat->rows;
    int N = mat->cols;
    dim3 threadSize(256);
    dim3 blockSize(N > 256 ? ceil(N / 256) : 1);
    mat_axis_sum_kernel<<<blockSize, threadSize>>>(mat->mat, out->mat, M, N);
    cudaDeviceSynchronize();
}