from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def main():
    in_size = 5  # length * width of image, but 1 for now
    num_Of_Input = 2 # number of images
    out_size = 2  # size of output
    layer_size = 1  # amount of neurons in the layer


    X = np.random.randint(0, 2, (in_size, num_Of_Input))  # input layer, tested with random int 0 to 1 for grayscale
    # Z = np.zeros((layer_size, 1))  # activation of neurons in layer
    W = np.random.randn(in_size, out_size)  # number of weights as input
    B = np.random.randn(layer_size, 1)
    Y = np.random.rand(out_size, num_Of_Input)  # output_layer_truths where rows are outputs and col are examples

    print("X: \n", X, "\n")
    print("W: \n", W.T, "\n")
    print("Y: \n", Y, "\n")
    #vectorized feedforward. W transpose weights * X training examples + B
    Z = np.dot(W.T, X) + B

    A = 1 / (1 + np.exp(-Z))
    print("A: \n", A)

    # Loss Function
    #i x j  matrix where i is number of outputs and j are examples
    J = -(Y * np.log(A) + ((1 - Y) * np.log(1-A)))
    print("J: \n", J, "\n")
    J = np.sum(J, axis=1, keepdims=True) / num_Of_Input
    # J = J.T # Make the output vertical
    print("J: \n", J, "\n")

if __name__ == '__main__':
    main()
