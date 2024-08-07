#ifndef LINEAR_H
#define LINEAR_H

#include "stdio.h"
#include "weight_update.h"
#include "matrix.h"
#include "cuda_runtime.h"

class Linear
{
private:
    size_t input_dim;
    size_t output_dim;

public:
    Matrix *output;
    Matrix *input;
    Matrix *weights;
    Matrix *biases;
    Matrix *grad_weights;
    Matrix *grad_biases;
    Matrix *grad_inputs;

public:
    Linear(size_t input_dim, size_t output_dim);
    Matrix *Linear::forward(Matrix *input);
    void Linear::backward(Matrix *grad);
    void update_weights(float lr);
    ~Linear();
};

#endif