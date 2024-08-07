#include "linear.h"

Linear::Linear(int input_dim, int output_dim)
{
    weights = new Matrix(input_dim, output_dim);
    biases = new Matrix(1, output_dim);
}

Matrix *Linear::forward(Matrix *input)
{
    this->input = input;
    output = input->dot(weights);
    output = output->add(biases);
    return output;
}

void Linear::backward(Matrix *grads)
{
    grad_weights = (mat_transpose(input))->dot(grads);
    grad_inputs = (grads)->dot(mat_transpose(weights));
    grad_biases = mat_axis_sum(grads, 0);
}

void Linear::update_weights(float lr)
{
    update_parameters(weights, grad_weights, lr);
    update_parameters(biases, grad_biases, lr);
}

Linear::~Linear()
{
    delete (weights);
    delete (biases);
    delete (input);
    delete (output);
    delete (grad_weights);
    delete (grad_biases);
    delete (grad_inputs);
}
