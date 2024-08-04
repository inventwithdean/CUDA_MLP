#include "linear.h"
#include "relu.h"
#include "stdio.h"
#include "cuda_runtime.h"

void printMatrix(float *matrix, int M, int N);
void InitializeMatrixRandom(float *matrix, int M, int N);

int main()
{
    size_t batch_size = 128;
    size_t input_dim = 768;
    size_t output_dim = 32;

    float *input; // TEST SIZE: (8, 128)
    size_t input_size = batch_size * input_dim * sizeof(float);
    cudaError_t err = cudaMallocManaged(&input, input_size);
    if (err != cudaSuccess)
    {
        printf("cudaMallocManaged Failed: %s\n", cudaGetErrorString(err));
    }
    InitializeMatrixRandom(input, batch_size, input_dim);

    Linear linear1 = Linear(input_dim, output_dim, batch_size); // 128 by 64 -> 32 by 64
    ReLU relu1 = ReLU(batch_size, output_dim);
    Linear linear2 = Linear(output_dim, 1, batch_size); // 32 by 64 -> 32 by 1
    ReLU relu2 = ReLU(batch_size, 1);

    // FORWARD PASS
    float *out = linear1.forward(input);
    out = relu1.forward(out);
    out = linear2.forward(out);
    out = relu2.forward(out);
    cudaFree(input);
    return 0;
}

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
            float random = rand() % 5;
            matrix[i * N + j] = random >= 3 ? -random : random;
        }
    }
}