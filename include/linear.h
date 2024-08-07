#ifndef LINEAR_H
#define LINEAR_H

#include "matmul.h"
#include "mat_axis_sum.h"
#include "mat_transpose.h"
#include "stdio.h"
#include "weight_update.h"
#include "mat_add.h"
#include "matrix.h"
#include "cuda_runtime.h"

class Linear
{
private:
public:
    Matrix *output;
    Matrix *input;
    Matrix *weights;
    Matrix *biases;
    Matrix *grad_weights;
    Matrix *grad_biases;
    Matrix *grad_inputs;

public:
    Linear(int input_dim, int output_dim);
    Matrix *Linear::forward(Matrix *input);
    void Linear::backward(Matrix *grad);
    void update_weights(float lr);
    ~Linear();
};

#endif