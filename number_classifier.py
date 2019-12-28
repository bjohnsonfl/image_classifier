from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def main():
    in_size = 5  # length * width of image, but 1 for now
    num_Of_Input = 2 # number of images
    out_size = 2  # size of output
    step_Size = 0.1 # learning rate
    iter = 100 # iterations of gradient decent


    X = np.random.randint(0, 2, (in_size, num_Of_Input))  # input layer, tested with random int 0 to 1 for grayscale
    # Z = np.zeros((layer_size, 1))  # activation of neurons in layer
    W = np.random.randn(in_size, out_size)  # number of weights as input
    B = np.random.randn(out_size, 1)
    Y = np.random.rand(out_size, num_Of_Input)  # output_layer_truths where rows are outputs and col are examples

    print("X: \n", X, "\n")
    print("W: \n", W, "\n")
    print("B: \n", B, "\n")
    print("Y: \n", Y, "\n")

    for _ in range(0,iter):
        #vectorized feedforward. W transpose weights * X training examples + B
        Z = np.dot(W.T, X) + B

        A = 1 / (1 + np.exp(-Z))
        #print("A: \n", A)

        # Loss/Cost Function
        #i x j  matrix where i is number of outputs and j are examples
        J = -(Y * np.log(A) + ((1 - Y) * np.log(1-A)))
        #print("J: \n", J, "\n")
        J = np.sum(J, axis=1, keepdims=True) / num_Of_Input
        # J = J.T # Make the output vertical
        print("J: \n", J, "\n")

        # Gradient Decent Single Iteration
        dZ = A - Y
        #print("dZ: \n", dZ, "\n")
        dW = (1/num_Of_Input) * np.dot(X, dZ.T) # each column are for each additional neuron in the layer
        #print("dW: \n", dW, "\n")
        dB = (1/num_Of_Input) * np.sum(dZ, axis=1, keepdims=True)
        #print("dB: \n", dB, "\n")

        W = W - step_Size*dW
        #print("W: \n", W, "\n")

        B = B - step_Size*dB
        #print("B: \n", B, "\n")


if __name__ == '__main__':
    main()
