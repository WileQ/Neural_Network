# Neural Network from Scratch
A self-constructed neural network & test on minist_csv dataset to recognize handwritten digits without using TensorFlow or PyTorch.

# Features
The neural network is implemented with:
- Fully connected layers
- Sigmoid activation functions for hidden layers
- Softmax activation function for the output layer
- Cross-entropy loss as the cost function
- Gradient descent for backpropagation and parameter updates

# Dataset
This model uses 1% of the provided datasets with each line containing a csv notation of 28x28 pixel grayscale images forming a number.
- mnist_train.csv - training data
- mnist_test.csv - test data

# Usage
Clone the repository:
```bash
git clone https://github.com/WileQ/Neural_Network.git
cd Neural_Network
```
Run it:
```bash
python neural_network.py
```
