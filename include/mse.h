#ifndef MSE_H
#define MSE_H
#include "cuda_runtime.h"
#include "math.h"
#include "stdio.h"
class MeanSquaredError
{
private:
    int batch_size;
    float *grads;

public:
    MeanSquaredError(int batch_size);
    ~MeanSquaredError();
    float *CalculateGradients(float *outputs, float *targets);
    float CalculateLoss(float *outputs, float *targets);
};

#endif
