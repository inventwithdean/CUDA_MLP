This repository contains a pure CUDA C++ implementation of a Multilayer Perceptron (MLP) neural network with a Linear Regression Dataset with Apache 2 license on which the bnetwork attained a loss of 0.07. By building from the ground up, we gain a deep understanding of the inner workings of neural networks and the performance benefits of GPU acceleration.

![power](https://github.com/user-attachments/assets/c2c4428f-ae5c-4970-b707-4b856feef508)

## Disclaimer: 
* This project is for educational purposes and may not be optimized for production use.

## Features:

Pure CUDA C++ implementation: No external libraries or frameworks.
GPU acceleration: Leverage the power of GPUs for high performance.
Modular design: Clear separation of concerns for maintainability.
Detailed comments: Explanations for code clarity and understanding.
## Getting Started:
* Clone the repository:
Bash
`git clone https://github.com/inventwithdean/CUDA_MLP.git`

* Set up CUDA environment: Ensure you have a CUDA-capable GPU and the necessary CUDA toolkit installed.
* Compile the code: Use a CUDA-compatible compiler to build the project.
* Run the executable: Execute the generated binary to run the MLP.
## Structure:
* include: Header files for classes and functions along with some kernels for MSE and ReLUs.
* kernels: CUDA kernels for matrix related calculations
* main.cu: Main file where dataset is loaded and Optimization loop happens
* dataset.cu: A very simple and lightweight csv Dataset Loader class
## Future Improvements:

* Implementing Softmax and Convolutional layers.
* Add optimization techniques like momentum.
* Explore different activation functions and network architectures.
* Improve performance through kernel tuning and optimization.
## Contributing:

Feel free to contribute to this project by:

* Submitting bug reports
* Suggesting new features
* Improving the code
* Writing documentation
  
Let's dive into the world of deep learning together!


## License:
MIT
