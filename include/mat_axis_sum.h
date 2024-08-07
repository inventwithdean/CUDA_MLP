#ifndef MATAXIS_SUM_H
#define MATAXIS_SUM_H
#include "cuda_runtime.h"
#include "matrix.h"

Matrix *mat_axis_sum(Matrix *mat, int axis = 0);

#endif