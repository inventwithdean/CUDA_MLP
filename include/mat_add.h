#ifndef MATADD_H
#define MATADD_H

#include "stdio.h"
#include "cuda_runtime.h"

void mat_add_broadcasted(float *weights, float *biases, float *out, int M, int N);

#endif