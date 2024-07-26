import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("mnist_train.csv")
data = data.sample(frac=0.01)

Y = data.iloc[:, 0]
Y = np.array(Y)
Y = Y.reshape(-1, 1)

X = data.iloc[:, 1:]
X = np.array(X)
X = X/255

Y_one_hot = np.zeros((Y.size, 10))
Y_one_hot[np.arange(Y.size), Y.flatten()] = 1



data = pd.read_csv("mnist_test.csv")
data = data.sample(frac=0.01)

Y_test = data.iloc[:, 0]
Y_test = np.array(Y_test)
Y_test = Y_test.reshape(-1, 1)

X_test = data.iloc[:, 1:]
X_test = np.array(X_test)
X_test = X_test/255

Y_test_one_hot = np.zeros((Y_test.size, 10))
Y_test_one_hot[np.arange(Y_test.size), Y_test.flatten()] = 1


layers = [X.shape[1], 16, 10]
layer_length = len(layers)

learning_rate = 0.01
iterations = 20000

class Neural_network():

    def initializing_parameters(self):
        weights = {}
        biases = {}

        for i in range(1, layer_length):
            weights[i] = np.random.rand(layers[i-1], layers[i]) * np.sqrt(2. / layers[i-1])
            biases[i] = np.random.rand(1, layers[i])


        return weights, biases

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=1, keepdims=True)

    def forward_propagation(self, X, weights, biases):
        A = X
        cache_A = {}
        cache_Z = {}

        for i in range(1, layer_length):
            A_previous = A
            Z = np.dot(A_previous, weights[i]) + biases[i]
            if i == layer_length - 1:
                A = self.softmax(Z)
            else:
                A = 1 / (1 + np.exp(-Z))
            cache_A[i-1] = A_previous
            cache_Z[i] = Z

        cache_A[layer_length - 1] = A
        return A, cache_A, cache_Z


    def backward_propagation(self, X, Y, weights, cache_A):
        gradients_weights = {}
        gradients_biases = {}
        m = X.shape[0]
        dA = cache_A[layer_length - 1] - Y

        for i in range(layer_length - 1, 0, -1):
            if i == layer_length - 1:
                dZ = dA
            else:
                dZ = dA * (cache_A[i] * (1 - cache_A[i]))
            dW = np.dot(np.transpose(cache_A[i - 1]), dZ) / m
            dB = np.sum(dZ, axis=0, keepdims=True) / m
            dA = np.dot(dZ, np.transpose(weights[i]))

            gradients_weights[i] = dW
            gradients_biases[i] = dB

        return gradients_weights, gradients_biases


    def updating_parameters(self, weights, biases, gradients_weights, gradients_biases, learning_rate):
        for i in range(1, layer_length):
            weights[i] -= learning_rate * gradients_weights[i]
            biases[i] -= learning_rate * gradients_biases[i]

        return weights, biases

    def compute_loss(self, Y, A):
        m = Y.shape[0]
        loss = -np.sum(Y * np.log(A)) / m
        return loss

    def training(self, X, Y, learning_rate, iterations):
        weights, biases = self.initializing_parameters()
        losses = []

        for i in range(iterations):
            A, cache_A, cache_Z = self.forward_propagation(X, weights, biases)
            loss = self.compute_loss(Y, A)
            losses.append(loss)
            gradients_weights, gradients_biases = self.backward_propagation(X, Y, weights, cache_A)
            weights, biases = self.updating_parameters(weights, biases, gradients_weights, gradients_biases, learning_rate)

            if i % 1000 == 0:
                print(f"Iteration {i}, Loss: {loss}")

        return weights, biases, losses

    def plot_losses(self, losses):
        plt.plot(losses)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss over Iterations')
        plt.show()

    def predict(self, X, weights, biases):
        A, _, _ = self.forward_propagation(X, weights, biases)
        return np.argmax(A, axis=1)


Nn = Neural_network()
weights, biases, losses = Nn.training(X, Y_one_hot, learning_rate, iterations)
Nn.plot_losses(losses)

predictions = Nn.predict(X_test, weights, biases)





for i in range(len(predictions)):
    print(predictions[i], Y_test[i])

accuracy = np.mean(predictions == Y_test.flatten())
print(f"Test Accuracy: {accuracy * 100:.2f}%")

