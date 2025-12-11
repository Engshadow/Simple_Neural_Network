import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)
def linear(x):
    return x

def linear_derivative(x):
    return 1.0



class SimpleNeuralNetwork:
    def __init__(self, layer_sizes, activations):

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.weights = []
        self.biases = []
        
        # Initialize Weights and Biases randonmly
        for i in range(len(layer_sizes) - 1):

            scale = np.sqrt(2.0 / layer_sizes[i]) 
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def get_activation_func(self, name):
        if name == 'sigmoid':
            return sigmoid, sigmoid_derivative
        elif name == 'relu':
            return relu, relu_derivative
        else:
            return linear, linear_derivative

    def forward(self, X):
        self.layer_inputs = []
        self.layer_activations = [X]

        activation_output = X

        for layer_index in range(len(self.weights)):
            W = self.weights[layer_index]
            b = self.biases[layer_index]
            act_func, _ = self.get_activation_func(self.activations[layer_index])

            Z = np.dot(activation_output, W) + b
            self.layer_inputs.append(Z)

            activation_output = act_func(Z)
            self.layer_activations.append(activation_output)

        return activation_output


    def backward(self, X, y, learning_rate):

        m = X.shape[0]
        y_pred = self.layer_activations[-1]

        dA = 2 * (y_pred - y) / m
        for layer_index in reversed(range(len(self.weights))):

            Z = self.layer_inputs[layer_index]
            y_prev = self.layer_activations[layer_index]

            _, act_deriv = self.get_activation_func(self.activations[layer_index])
            
            dZ = dA * act_deriv(Z)

            # gradients
            dW = np.dot(y_prev.T, dZ) 
            dB = np.sum(dZ, axis=0, keepdims=True) 

            # backpropagate error
            dA = np.dot(dZ, self.weights[layer_index].T)

            # update weights
            self.weights[layer_index] -= learning_rate * dW
            self.biases[layer_index] -= learning_rate * dB
    
    def predict(self, X):
            probabilities = self.forward(X)
            predictions = (probabilities >= 0.5).astype(int)
            return predictions
