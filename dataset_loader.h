#ifndef DATASET_LOADER_H
#define DATASET_LOADER_H

#include "include/matrix.h"

class Dataset
{
private:
    Matrix *input_matrix;
    Matrix *target_matrix;

public:
    Dataset(int max_rows);
    void load_dataset(int max_rows = 256);
    Matrix *get_inputs();
    Matrix *get_targets();
    ~Dataset();
};

#endif