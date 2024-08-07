#include "stdio.h"
#include "include/linear.h"
#include "include/mse.h"

int main()
{
    srand(1337);

    Matrix *input = new Matrix(128, 1);
    Matrix *targets = new Matrix(128, 1);
    // Testing on {x * 5 - 15 : x belongs to 0 to 4}
    for (int i = 0; i < 128; i++)
    {
        input->mat[i] = (float)i;
        targets->mat[i] = (float)i * 5 - 15;
    }

    printf("Initial Parameters: \n");
    Linear linear = Linear(1, 1);
    MeanSquaredError mse = MeanSquaredError(4);

    // Optimization Loop
    for (int i = 0; i < 200; i++)
    {
        Matrix *out = linear.forward(input);

        mse.CalculateGradients(out, targets);
        float loss = mse.CalculateLoss(out, targets);
        if (i % 10 == 0)
            printf("Loss: %.2f\n", loss);

        linear.backward(mse.mse_grads);
        linear.update_weights(0.1);

        // Deleting the Matrices
        delete out;
    }

    printf("Trained Parameters: \n");
    linear.weights->print();
    linear.biases->print();

    delete input;
    delete targets;
    printf("No Errors yay!\n");
    return 0;
}
