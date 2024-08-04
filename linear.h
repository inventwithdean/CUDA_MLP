#ifndef LINEAR_H
#define LINEAR_H

#include "matmul.h"
#include "stdio.h"
#include "cuda_runtime.h"

class Linear
{
private:
    int input_dim;
    int output_dim;
    int batch_size;
    bool bias;
    float *output;

public:
    float *weights;

public:
    Linear(int input_dim, int output_dim, int batch_size = 32, bool bias = false);
    float *forward(float *input);
    ~Linear();
};

#endif