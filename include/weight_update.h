#ifndef VANILLA_GD_H
#define VANILLA_GD_H
#include "cuda_runtime.h"

void update_weights(float *weights, float *grads, int M, int N, float learning_rate);
#endif
