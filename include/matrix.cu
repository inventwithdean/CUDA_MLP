#include "matrix.h"

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols)
{
    size_t size = rows * cols * sizeof(float);
    cudaError_t err = cudaMallocManaged(&mat, size);
    if (err != cudaSuccess)
    {
        printf("cudaMallocManaged Failed: %s\n", cudaGetErrorString(err));
    }
    FillRandom();
}

Matrix *Matrix::dot(Matrix *other)
{
    // This multiplied by mat
    size_t out_rows = rows;
    size_t out_cols = other->cols;
    Matrix *out = new Matrix(out_rows, out_cols);
    matmul(this->mat, other->mat, out->mat, rows, cols, out_cols);
    return out;
}

Matrix *Matrix::add(Matrix *other)
{
    // Currently, only supports axis 0 broadcasting!
    // In the form of weights(M, N) + biases (1, N)
    size_t out_rows = rows;
    size_t out_cols = cols;
    Matrix *out = new Matrix(rows, cols);
    mat_add_broadcasted(mat, other->mat, out->mat, out_rows, out_cols);
    return out;
}

void Matrix::print()
{
    printf("Matrix: Shape (%d, %d)\n", rows, cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int idx = i * cols + j;
            printf("%f ", mat[idx]);
        }
        printf("\n");
    }
    printf("\n");
}

void Matrix::FillRandom()
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int idx = i * cols + j;
            mat[idx] = (float)rand() / (float)(RAND_MAX / 5);
        }
    }
}

Matrix::~Matrix()
{
    cudaFree(mat);
}
