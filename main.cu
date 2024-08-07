#include "stdio.h"
#include "include/linear.h"
#include "include/mse.h"
#include "include/relu.h"
#include "dataset_loader.h"

// FIX MEMORY LEAKS! Currently, pretty much everything is leaking!
int main()
{
    srand(1337);
    size_t batch_size = 256 * 3;
    Dataset dataset = Dataset(batch_size);
    Matrix *input = dataset.get_inputs();
    Matrix *targets = dataset.get_targets();

    // Multilayer Perceptron
    Linear linear1 = Linear(5, 8);
    ReLU relu1 = ReLU();
    Linear linear2 = Linear(8, 16);
    ReLU relu2 = ReLU();
    Linear linear3 = Linear(16, 32);
    ReLU relu3 = ReLU();
    Linear linear4 = Linear(32, 1);

    MeanSquaredError mse = MeanSquaredError(batch_size);
    float learning_rate = 0.1;
    // Optimization Loop
    for (int i = 0; i < 100; i++)
    {
        Matrix *out = linear1.forward(input);
        out = relu1.forward(out);
        out = linear2.forward(out);
        out = relu2.forward(out);
        out = linear3.forward(out);
        out = relu3.forward(out);
        out = linear4.forward(out);

        mse.CalculateGradients(out, targets);
        float loss = mse.CalculateLoss(out, targets);
        printf("Loss: %.2f\n", loss);

        // linear2.backward(mse.mse_grads);
        linear4.backward(mse.mse_grads);

        relu3.backward(linear4.grad_inputs);
        linear3.backward(linear4.grad_inputs);

        relu2.backward(linear3.grad_inputs);
        linear2.backward(linear3.grad_inputs);

        relu1.backward(linear2.grad_inputs);
        linear1.backward(linear2.grad_inputs);

        linear4.update_weights(learning_rate);
        linear3.update_weights(learning_rate);
        linear2.update_weights(learning_rate);
        linear1.update_weights(learning_rate);
    }

    delete input;
    delete targets;
    return 0;
}
