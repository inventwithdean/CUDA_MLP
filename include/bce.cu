#include "bce.h"
#define EPSILON 1e-7

BinaryCrossEntropy::BinaryCrossEntropy(int batch_size) : batch_size(batch_size)
{
    size_t size_grads = batch_size * sizeof(float);
    cudaError_t err = cudaMallocManaged(&grads, size_grads);
    if (err != cudaSuccess)
    {
        printf("cudaMallocManaged Failed: %s", cudaGetErrorString(err));
    }
}

__global__ void BCE_kernel(float *outputs, float *targets, float *grads, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        if (outputs[idx] == 0) // Network predicted 0
        {
            grads[idx] = log(1 - targets[idx] + EPSILON) / N;
        }
        else // Network predicted 1
        {
            grads[idx] = -log(targets[idx] + EPSILON) / N;
        }
    }
}

float *BinaryCrossEntropy::CalculateGradients(float *outputs, float *targets)
{
    // TODO: IMPLEMENT Softmax and Gradients
    dim3 threadSize(256);
    dim3 numBlocks(ceil(float(batch_size) / 256));
    BCE_kernel<<<threadSize, numBlocks>>>(outputs, targets, grads, batch_size);
    cudaDeviceSynchronize();
    return grads;
}

BinaryCrossEntropy::~BinaryCrossEntropy()
{
    cudaFree(grads);
}