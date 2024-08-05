#ifndef RELU_H

#define RELU_H

#include "cuda_runtime.h"
#include "math.h"
class ReLU
{
private:
    float *output;
    // M BY N MATRIX
    int M;
    int N;

public:
    ReLU(int input_dim, int output_dim);
    float *forward(float *input);
    ~ReLU();
};

#endif