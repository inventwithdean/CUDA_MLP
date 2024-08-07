#ifndef MATRIX_H
#define MATRIX_H
#include "cuda_runtime.h"
#include "matmul.h"
#include "mat_add.h"
#include "stdio.h"

class Matrix
{
private:
public:
    int rows;
    int cols;
    float *mat;
    Matrix(int rows, int cols);
    Matrix *dot(Matrix *other); // Returns this matrix multiplied with mat
    Matrix *add(Matrix *other); // Returns this matrix added with mat
    void print();
    void FillRandom();
    ~Matrix();
};
#endif