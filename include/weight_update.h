#ifndef VANILLA_GD_H
#define VANILLA_GD_H
#include "cuda_runtime.h"
#include "matrix.h"

void update_parameters(Matrix *weight_matrix, Matrix *grad_matrix, float learning_rate);
#endif
