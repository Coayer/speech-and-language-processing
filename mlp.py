import numpy as np


class Layer:
    def __init__(self, size, prev_layer_size):
        self.size = size
        self.weights = np.zeros((prev_layer_size + 1, self.size))  # +1 size for bias

    def __repr__(self):
        return str(self.weights)

    def forward(self, input_x):
        x = np.append(input_x, -1)    # -1 input for bias
        output = np.zeros(self.size)

        for i in range(self.size):
            output[i] = self.activation_function(np.dot(x, self.weights[:, i]))

        return output

    def activation_function(self, z):
        return (1 + 2.7182 ** -z) ** -1


class NeuralNetwork:
    def __init__(self, l_sizes):
        self.layers = [Layer(l_sizes[i], l_sizes[i - 1]) for i in range(1, len(l_sizes))]

    def forward_propagate(self, vector):
        for layer in self.layers:
            vector = layer.forward(vector)

        return vector


test = NeuralNetwork([2, 2])
print(test.forward_propagate([1, 0]))
