#include "linear.h"

Linear::Linear(size_t input_dim, size_t output_dim) : input_dim(input_dim), output_dim(output_dim)
{
    weights = new Matrix(input_dim, output_dim);
    biases = new Matrix(1, output_dim);
    output = nullptr;
    grad_inputs = nullptr;
    grad_biases = new Matrix(1, output_dim);
    grad_weights = new Matrix(input_dim, output_dim);
}

Matrix *Linear::forward(Matrix *input)
{
    if (output == nullptr)
    {
        output = new Matrix(input->rows, weights->cols);
    }
    if (grad_inputs == nullptr)
    {
        grad_inputs = new Matrix(input->rows, input->cols);
    }
    this->input = input;
    Matrix out_temp = Matrix(input->rows, weights->cols);
    input->dot(weights, &out_temp);
    out_temp.add(biases, output);
    return output;
}

void Linear::backward(Matrix *grads)
{
    Matrix transposed_input = Matrix(input->cols, input->rows);
    input->transpose(&transposed_input);
    transposed_input.dot(grads, grad_weights);
    Matrix transposed_weights = Matrix(output_dim, input_dim);
    weights->transpose(&transposed_weights);
    grads->dot(&transposed_weights, grad_inputs);

    grads->sum(grad_biases, 0);
}

void Linear::update_weights(float lr)
{
    update_parameters(weights, grad_weights, lr);
    update_parameters(biases, grad_biases, lr);
}

Linear::~Linear()
{
    delete (input);
    delete (output);
    delete (weights);
    delete (biases);
    delete (grad_weights);
    delete (grad_biases);
    delete (grad_inputs);
}
