from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def main():
    in_size = 5  # length * width of image, but 1 for now
    num_Of_Input = 2 # number of images
    out_size = 1  # size of output
    layer_size = 1  # amount of neurons in the layer


    X = np.random.randint(0, 2, (in_size, num_Of_Input))  # input layer, tested with random int 0 to 1 for grayscale
    # Z = np.zeros((layer_size, 1))  # activation of neurons in layer
    W = np.random.randn(in_size, 1)  # number of weights as input
    B = np.random.randn(layer_size,1)
    #Y = np.zeros((out_size, 1))  # output_layer

    print("X: \n", X, "\n")
    print("W: \n", W.T, "\n")


    Z = np.dot(W.T, X) + B

    Y = 1 / (1 + np.exp(-Z))
    print("Y: \n", Y)


if __name__ == '__main__':
    main()
