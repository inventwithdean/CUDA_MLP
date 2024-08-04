#include "linear.h"

Linear::Linear(int input_dim, int output_dim, int batch_size, bool bias) : input_dim(input_dim), output_dim(output_dim), batch_size(batch_size), bias(bias)
{
    // Currently, bias is not supported!
    // Only floats are supported
    size_t weight_size = input_dim * output_dim * sizeof(float);
    cudaError_t err;

    // Allocating Memory for Weight Matrix
    err = cudaMallocManaged(&weights, weight_size);
    if (err != cudaSuccess)
    {
        printf("cudaMallocManaged Failed: %s", cudaGetErrorString(err));
    }
    // Allocating Memory for Output Matrix
    size_t output_size = batch_size * output_dim * sizeof(float);
    err = cudaMallocManaged(&(this->output), output_size);
    if (err != cudaSuccess)
    {
        printf("cudaMallocManaged Failed: %s", cudaGetErrorString(err));
    }

    // Initializing weights with random Values
    for (int i = 0; i < input_dim; i++)
    {
        for (int j = 0; j < output_dim; j++)
        {
            float random = rand() % 5;
            weights[i * output_dim + j] = random >= 3 ? -random : random;
        }
    }
}

float *Linear::forward(float *input)
{
    matmul(input, weights, output, batch_size, input_dim, output_dim);
    return this->output;
}

Linear::~Linear()
{
    cudaFree(weights);
    cudaFree(output);
}
