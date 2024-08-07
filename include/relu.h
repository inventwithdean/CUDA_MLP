#ifndef RELU_H

#define RELU_H

#include "cuda_runtime.h"
#include "math.h"
#include "matrix.h"

class ReLU
{
private:
    Matrix *input;
    Matrix *output;

public:
    ReLU();
    Matrix *forward(Matrix *input);
    void backward(Matrix *grads);
    ~ReLU();
};

#endif