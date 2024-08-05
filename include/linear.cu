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
    // Allocating Memory for Weight Gradients
    err = cudaMallocManaged(&grad_weights, weight_size);
    if (err != cudaSuccess)
    {
        printf("cudaMallocManaged Failed: %s", cudaGetErrorString(err));
    }
    // Allocating Memory for Input Cache
    size_t input_size = input_dim * batch_size * sizeof(float);
    err = cudaMallocManaged(&input, input_size);
    if (err != cudaSuccess)
    {
        printf("cudaMallocManaged Failed: %s", cudaGetErrorString(err));
    }
    // Allocating Memory for Output Matrix
    size_t output_size = batch_size * output_dim * sizeof(float);
    err = cudaMallocManaged(&output, output_size);
    if (err != cudaSuccess)
    {
        printf("cudaMallocManaged Failed: %s", cudaGetErrorString(err));
    }

    // Initializing weights with random Values
    for (int i = 0; i < input_dim; i++)
    {
        for (int j = 0; j < output_dim; j++)
        {
            weights[i * output_dim + j] = rand() % 5;
        }
    }
}

float *Linear::forward(float *input)
{
    size_t sizecpy = input_dim * batch_size * sizeof(float);
    cudaMemcpy(this->input, input, sizecpy, cudaMemcpyHostToDevice);
    matmul(input, weights, output, batch_size, input_dim, output_dim);
    return this->output;
}

void Linear::backward(float *grad)
{
    // Grad Shape -> Out Shape (batch_size, output_dim)
    float *transposed_mat = mat_transpose(input, batch_size, input_dim); // DON'T FORGET TO FREE THIS transposed_mat
    // Multiply grad by input's transpose
    matmul(transposed_mat, grad, grad_weights, input_dim, batch_size, output_dim);
    cudaFree(transposed_mat);
}

Linear::~Linear()
{
    cudaFree(weights);
    cudaFree(grad_weights);
    cudaFree(output);
    cudaFree(input);
}
