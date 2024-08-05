#ifndef BCE_H
#define BCE_H
#include "cuda_runtime.h"
#include "math.h"
#include "stdio.h"
class BinaryCrossEntropy
{
private:
    int batch_size;
    float *grads;

public:
    BinaryCrossEntropy(int batch_size);
    ~BinaryCrossEntropy();
    float *CalculateGradients(float *outputs, float *targets);
};

#endif
