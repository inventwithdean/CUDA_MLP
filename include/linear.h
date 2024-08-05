#ifndef LINEAR_H
#define LINEAR_H

#include "matmul.h"
#include "mat_transpose.h"
#include "stdio.h"
#include "cuda_runtime.h"

class Linear
{
private:
    int input_dim;
    int output_dim;
    int batch_size;
    bool bias;

public:
    float *output;
    float *input;
    float *weights;
    float *grad_weights;

public:
    Linear(int input_dim, int output_dim, int batch_size = 32, bool bias = false);
    float *forward(float *input);
    void backward(float *grad);
    ~Linear();
};

#endif