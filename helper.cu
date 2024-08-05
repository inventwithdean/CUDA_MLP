#include "helper.h"

void printMatrix(float *matrix, int M, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%.2f ", matrix[i * N + j]);
        }
        printf("\n");
    }
}

void InitializeMatrixRandom(float *matrix, int M, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i * N + j] = rand() % 5;
        }
    }
}