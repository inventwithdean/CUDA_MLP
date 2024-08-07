#include "dataset_loader.h"
#include "fstream"
#include "iostream"
#include "sstream"
using namespace std;

Dataset::Dataset(int max_rows)
{
    load_dataset(max_rows);
}

void Dataset::load_dataset(int max_rows)
{
    int MAX_ROWS = max_rows;
    input_matrix = new Matrix(MAX_ROWS, 5); // Not loading last 2 columns
    target_matrix = new Matrix(MAX_ROWS, 1);
    ifstream file("Mobile-Price-Prediction-cleaned_data.csv");
    if (!file.is_open())
    {
        printf("Error opening dataset file!\n");
    }
    string line;
    getline(file, line);
    int row = 0;
    while (getline(file, line))
    {
        if (row >= MAX_ROWS)
            break;
        // cout << line << endl;
        stringstream ss(line);
        string cell;
        int col = -1;
        while (getline(ss, cell, ','))
        {
            if (col == -1)
            {
                // RATINGS
                target_matrix->mat[row] = stof(cell);
            }
            else
            {
                // FEATURES
                int idx = row * 5 + col;
                input_matrix->mat[idx] = stof(cell);
            }
            col++;
        }
        row++;
    }

    file.close();

    // Printing Loaded Dataset
    // input_matrix->print();
    // target_matrix->print();
}

Matrix *Dataset::get_inputs()
{
    return input_matrix;
}

Matrix *Dataset::get_targets()
{
    return target_matrix;
}

Dataset::~Dataset()
{
    delete input_matrix;
    delete target_matrix;
}
