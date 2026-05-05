# import numpy as np

# inputs = [[1, 2, 3, 2.5], 
#           [2.0, 5.0, -1.0, 2.0], 
#           [-1.5, 2.7, 3.3, -0.8]]

# weights = [[0.2, 0.8, -0.5, 1.0],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]]

# bias = [2, 3, 0.5]

# weights2 = [[0.1, -0.14, 0.5],
#             [-0.5, 0.12, -0.33],  
#             [-0.44, 0.73, -0.13]]

# bias2 = [-1, 2, -0.5]

# layer1_output = np.dot(inputs, np.array(weights).T) + bias # transpose the weights to match the dimensions for dot product

# layer2_output = np.dot(layer1_output, np.array(weights2).T) + bias2

# print(layer2_output) 

import numpy as np

np.random.seed(0) # set seed for reproducibility

X = [[1, 2, 3, 2.5], 
    [2.0, 5.0, -1.0, 2.0], 
    [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # initialize weights with small random values
        self.biases = np.zeros((1, n_neurons)) # initialize biases to zero

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases # calculate output of the layer

layer1 = Layer_Dense(4, 5) # create first layer with 4 inputs and 5 neurons

layer2 = Layer_Dense(5, 2) # create second layer with 5 inputs and 2 neurons

layer1.forward(X) # perform forward pass through the first layer
#print(layer1.output) # print the output of the first layer
layer2.forward(layer1.output) # perform forward pass through the second layer using the output of the first layer
print(layer2.output) # print the output of the second layer