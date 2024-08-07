#ifndef MATRIX_CALC_H
#define MATRIX_CALC_H
#include "cuda_runtime.h"
#include "../include/matrix.h"

class Matrix; // Forward declaration is very important here somehow!
void matmul(float *A, float *B, float *C, int M, int N, int P);
void transpose_matrix(Matrix *mat, Matrix *out);
void mat_add_broadcasted(float *weights, float *biases, float *out, int M, int N);
void mat_axis_sum(Matrix *mat, Matrix *out, int axis = 0);

#endif