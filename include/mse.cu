#include "mse.h"

MeanSquaredError::MeanSquaredError(int batch_size) : batch_size(batch_size)
{
    size_t size_grads = batch_size * sizeof(float);

    cudaMallocManaged(&grads, size_grads);
}

__global__ void MSE_kernel(float *outputs, float *targets, float *grads, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        grads[idx] = (outputs[idx] - targets[idx]) / N;
    }
}

float *MeanSquaredError::CalculateGradients(float *outputs, float *targets)
{
    dim3 threadSize(256);
    dim3 numBlocks(ceil(float(batch_size) / 256));
    MSE_kernel<<<threadSize, numBlocks>>>(outputs, targets, grads, batch_size);
    cudaDeviceSynchronize();
    return grads;
}

// CURRENTY RUNS ON CPU
float MeanSquaredError::CalculateLoss(float *outputs, float *targets)
{

    float loss = 0;
    for (int i = 0; i < batch_size; i++)
    {
        float diff = targets[i] - outputs[i];
        loss += (diff * diff) / batch_size;
    }
    loss /= 2;
    return loss;
}

MeanSquaredError::~MeanSquaredError()
{
    cudaFree(grads);
}