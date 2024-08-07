#ifndef MSE_H
#define MSE_H
#include "cuda_runtime.h"
#include "math.h"
#include "stdio.h"
#include "matrix.h"

class MeanSquaredError
{
private:
    int batch_size;

public:
    Matrix *mse_grads;
    MeanSquaredError(int batch_size);
    ~MeanSquaredError();
    Matrix *CalculateGradients(Matrix *output_matrix, Matrix *target_matrix);
    float CalculateLoss(Matrix *output_matrix, Matrix *target_matrix);
};

#endif
