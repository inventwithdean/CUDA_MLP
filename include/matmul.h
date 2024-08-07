#ifndef MATMUL_H
#define MATMUL_H

#include "cuda_runtime.h"
#include "math.h"

void matmul(float *A, float *B, float *C, int M, int N, int P);

#endif