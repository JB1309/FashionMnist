# FashionMnist
This repository implements a Feedforward Neural Network (FNN) using PyTorch to classify handwritten digits from the MNIST dataset. The model is designed to be flexible with multiple configurations, enabling experimentation with various hyperparameters to optimize model performance.


This project implements a Feedforward Neural Network to classify images from the MNIST dataset using a fully connected neural network. The goal is to evaluate different hyperparameter configurations and choose the best performing one based on validation accuracy.

Key Features:
Flexible Architecture: Adjust the number of hidden layers, units per layer, and choice of activation functions.
Multiple Optimizers: Support for different optimization algorithms like SGD, Momentum, and Adam.
Cross-Entropy Loss: Loss function used for training, suitable for multi-class classification.
Confusion Matrix: Provides insights into misclassifications after evaluation.
Installation
To use this project, you'll need Python 3.x and PyTorch. You can install the required dependencies using the following commands:

Install Dependencies:
pip install torch torchvision matplotlib seaborn scikit-learn


Load the MNIST dataset.
Preprocess the images.
Define and train the feedforward neural network.
Display validation accuracy after each epoch.
Evaluate the final model on the test dataset.
Plot the confusion matrix and show the loss comparison.
Model Configuration
This Feedforward Neural Network is configurable with various hyperparameters. The model's architecture can be modified by changing the following parameters:

Hyperparameters:
Input Size: The input size is fixed at 28 * 28 = 784 for MNIST images.
Hidden Layers: The number of hidden layers in the network.
Hidden Units: The number of neurons in each hidden layer.
Activation Function: You can choose between ReLU and Sigmoid.
Weight Initialization: Use Xavier or Random initialization for the network weights.
Optimizer: Available optimizers include SGD, Momentum, and Adam.
Epochs: Number of training epochs (default: 10).
Batch Size: Batch size for training (default: 64).
Learning Rate: The learning rate for the optimizer (default: 1e-3).
Example configuration:

python
Copy
input_size = 28 * 28  # Image size: 28x28 pixels
output_size = 10  # Number of classes in MNIST (digits 0-9)
hidden_layers = 3  # Number of hidden layers
hidden_units = 128  # Number of neurons in each hidden layer
epochs = 10  # Number of training epochs
learning_rate = 1e-3  # Learning rate for the optimizer
optimizer_choice = 'adam'  # Optimizer choice: 'sgd', 'momentum', 'adam'
activation = 'relu'  # Activation function choice: 'relu', 'sigmoid'
weight_init = 'xavier'  # Weight initialization choice: 'random', 'xavier'
Training
The model is trained using the Adam optimizer by default. The training process includes:

Data Loading: The MNIST dataset is loaded and split into training (90%) and validation (10%) sets.
Model Training: The Feedforward Neural Network is trained for the specified number of epochs (default: 10). During training, the model's performance is evaluated on the validation set.
Hyperparameter Tuning: Experiment with different configurations for optimal performance.
Example Training Log:
yaml
Copy
Epoch [1/10], Loss: 0.3841, Validation Accuracy: 87.50%
Epoch [2/10], Loss: 0.2798, Validation Accuracy: 87.70%
...
Epoch [10/10], Loss: 0.1864, Validation Accuracy: 87.95%
