#ifndef MATRIX_H
#define MATRIX_H
#include "cuda_runtime.h"
#include "../kernels/matrix_calc.h"
#include "stdio.h"

class Matrix
{
private:
public:
    int rows;
    int cols;
    float *mat;
    Matrix(int rows, int cols);
    void dot(Matrix *other, Matrix *out); // Returns this matrix multiplied with mat
    void add(Matrix *other, Matrix *out); // Returns this matrix added with mat
    void transpose(Matrix *out);          // Returns transpose of this matrix
    void sum(Matrix *out, int axis = 0);
    void print();
    void FillRandom();
    ~Matrix();
};
#endif