#include "mse.h"

MeanSquaredError::MeanSquaredError(int batch_size) : batch_size(batch_size)
{
    mse_grads = new Matrix(batch_size, 1);
}

__global__ void MSE_kernel(float *outputs, float *targets, float *grads, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        grads[idx] = (outputs[idx] - targets[idx]) / N;
    }
}

Matrix *MeanSquaredError::CalculateGradients(Matrix *output_matrix, Matrix *target_matrix)
{
    float *outputs = output_matrix->mat;
    float *targets = target_matrix->mat;
    dim3 threadSize(256);
    dim3 numBlocks(ceil(float(batch_size) / 256));
    MSE_kernel<<<threadSize, numBlocks>>>(outputs, targets, mse_grads->mat, batch_size);
    cudaDeviceSynchronize();
    return mse_grads;
}

// CURRENTY RUNS ON CPU
float MeanSquaredError::CalculateLoss(Matrix *output_matrix, Matrix *target_matrix)
{
    float *outputs = output_matrix->mat;
    float *targets = target_matrix->mat;
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
    delete (mse_grads);
}