#include "include/linear.h"
#include "include/relu.h"
#include "include/mse.h"
#include "include/weight_update.h"
#include "stdio.h"
#include "cuda_runtime.h"
#include "helper.h"

int main()
{
    size_t batch_size = 8;
    size_t input_dim = 1;
    size_t output_dim = 1;

    float *input;
    float *targets;
    cudaError_t err;

    // Calculating sizes of input and target matrices
    size_t input_size = batch_size * input_dim * sizeof(float);
    size_t target_size = batch_size * sizeof(float);

    // Initializing Memory for Inputs
    err = cudaMallocManaged(&input, input_size);
    if (err != cudaSuccess)
    {
        printf("cudaMallocManaged Failed: %s\n", cudaGetErrorString(err));
    }
    // Initializing Memory for Targets
    err = cudaMallocManaged(&targets, target_size);
    if (err != cudaSuccess)
    {
        printf("cudaMallocManaged Failed: %s\n", cudaGetErrorString(err));
    }

    // Initializing input and targets to Linear Function
    for (int i = 0; i < batch_size; i++)
    {
        input[i] = i;
        targets[i] = -i;
    };

    Linear linear = Linear(input_dim, output_dim, batch_size); // 4 by 8 -> 4 by 4

    MeanSquaredError mse = MeanSquaredError(batch_size);

    // Optimization Loop
    // print statements are for debugging purposes
    for (int i = 0; i < 15; i++)
    {
        // printf("Linear Layer Weights\n");
        // printMatrix(linear.weights, input_dim, output_dim);
        float *out = linear.forward(input);
        // printf("Linear Layer Output\n");
        // printMatrix(out, batch_size, output_dim);
        float loss = mse.CalculateLoss(out, targets);
        printf("Loss | %dth iteration: %.2f\n", i + 1, loss);
        float *grads = mse.CalculateGradients(out, targets);
        // printf("MSE Gradient\n");
        // printMatrix(grads, batch_size, output_dim);
        linear.backward(grads);
        // printf("Linear Layer Gradient\n");
        // printMatrix(linear.grad_weights, input_dim, output_dim);
        update_weights(linear.weights, linear.grad_weights, input_dim, output_dim, 0.1);
        // printf("Linear Layer Weights\n");
        // printMatrix(linear.weights, input_dim, output_dim);
        // if (i == 3)
        //     break;
    }

    cudaFree(input);
    cudaFree(targets);
    return 0;
}
