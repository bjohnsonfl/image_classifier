from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def logReg():
    in_size = 5  # length * width of image, but 1 for now
    num_Of_Input = 3  # number of images
    out_size = 2  # size of output
    step_Size = 0.1  # learning rate
    iter = 1  # iterations of gradient decent

    X = np.random.randint(0, 2, (in_size, num_Of_Input))  # input layer, tested with random int 0 to 1 for grayscale
    W = np.random.randn(in_size, out_size)  # number of weights as input
    B = np.random.randn(out_size, 1)
    Y = np.random.rand(out_size, num_Of_Input)  # output_layer_truths where rows are outputs and col are examples

    for _ in range(0, iter):
        # vectorized feedforward. W transpose weights * X training examples + B
        Z = np.dot(W.T, X) + B
        A = 1 / (1 + np.exp(-Z))

        # Loss/Cost Function
        # i x j  matrix where i is number of outputs and j are examples
        J = -(Y * np.log(A) + ((1 - Y) * np.log(1 - A)))
        J = np.sum(J, axis=1, keepdims=True) / num_Of_Input
        # J = J.T # Make the output vertical

        # Gradient Decent Single Iteration
        dZ = A - Y  # only for output layer. dz1 = (W2.T * dz2) .* g1'(z1) where g(z) is activation func
        dW = (1 / num_Of_Input) * np.dot(X, dZ.T)  # each column are for each additional neuron in the layer
        dB = (1 / num_Of_Input) * np.sum(dZ, axis=1, keepdims=True)

        W = W - step_Size * dW
        B = B - step_Size * dB


class Layer:
    """
    A layer initial parameters will dictate the size of the layer, number of examples,
    and in the future, the type of activation function.

    The layer block diagram will consist of the following:
        input: A[n-1]  (internal signal will be X)
        output: A[n] ((ixj) matrix where i is size of layer and j is number of examples
        feedback: dZ[n], W[n]  (this is used for gradient decent in layer [n-1])

        internal variables
            W, a (ixj) matrix where i is the size of layer n-1 and j is the size of the layer n
            B, a (ix1) matrix where i is the size of the layer n
            dZ, a (ixj) matrix where i is the size of the layer n and j are the number of examples
            dW, a (ixj) matrix where i is the size of layer n-1 and j is the size of layer n
            dB, a (ixj) matrix where i is the size of the layer n

            dZ dW and dB are the gradients of Z W and B which are used to with the learning rate to step to a min.
    """

    def __init__(self, in_size, out_size, num_Of_Examples):
        self.num_Of_Input = num_Of_Examples
        self.W = np.random.randn(in_size, out_size)
        self.B = np.random.randn(out_size, 1)
        # A and dZ need to be returnable for feed forward and feedback
        self.A = np.zeros((out_size, num_Of_Examples))
        self.dZ = np.zeros((out_size, num_Of_Examples))

    def forward(self, X):
        # vectorized feedforward. W transpose weights * X training examples + B
        Z = np.dot(self.W.T, X) + self.B
        self.A = 1 / (1 + np.exp(-Z))

        """ This is only for the last layer
        # Loss/Cost Function
        # i x j  matrix where i is number of outputs and j are examples
        J = -(Y * np.log(A) + ((1 - Y) * np.log(1 - A)))
        J = np.sum(J, axis=1, keepdims=True) / self.num_Of_Input
        # J = J.T # Make the output vertical
        """


class OutputLayer(Layer):
    def __init__(self, in_size, out_size, num_Of_Examples, Y):
        super().__init__(self, in_size, out_size, num_Of_Examples)
        self.Y = Y
        self.Jsum = np.zeros((out_size, 1))

    def forward(self, X):
        # vectorized feedforward. W transpose weights * X training examples + B
        Z = np.dot(self.W.T, X) + self.B
        self.A = 1 / (1 + np.exp(-Z))

        # This is only for the last layer
        # Loss/Cost Function
        # i x j  matrix where i is number of outputs and j are examples
        J = -(self.Y * np.log(self.A) + ((1 - self.Y) * np.log(1 - self.A)))
        J = np.sum(J, axis=1, keepdims=True) / self.num_Of_Input
        self.Jsum = J.T  # Make the output vertical


def main():
    test = Layer(5, 2, 3)
    in_size = 5  # length * width of image, but 1 for now
    num_Of_Input = 3  # number of images
    out_size = 2  # size of output
    step_Size = 0.1  # learning rate
    iter = 1  # iterations of gradient decent

    X = np.random.randint(0, 2, (in_size, num_Of_Input))  # input layer, tested with random int 0 to 1 for grayscale
    # Z = np.zeros((layer_size, 1))  # activation of neurons in layer
    W = np.random.randn(in_size, out_size)  # number of weights as input
    B = np.random.randn(out_size, 1)
    Y = np.random.rand(out_size, num_Of_Input)  # output_layer_truths where rows are outputs and col are examples

    print("X: \n", X, "\n")
    print("W: \n", W, "\n")
    print("B: \n", B, "\n")
    print("Y: \n", Y, "\n")

    for _ in range(0, iter):
        # vectorized feedforward. W transpose weights * X training examples + B
        Z = np.dot(W.T, X) + B

        A = 1 / (1 + np.exp(-Z))
        print("A: \n", A)

        # Loss/Cost Function
        # i x j  matrix where i is number of outputs and j are examples
        J = -(Y * np.log(A) + ((1 - Y) * np.log(1 - A)))
        # print("J: \n", J, "\n")
        J = np.sum(J, axis=1, keepdims=True) / num_Of_Input
        # J = J.T # Make the output vertical
        print("J: \n", J, "\n")

        # Gradient Decent Single Iteration
        dZ = A - Y  # only for output layer. dz1 = (W2.T * dz2) .* g1'(z1) where g(z) is activation func
        print("dZ: \n", dZ, "\n")
        print("X: \n", X, "\n")
        print("XT: \n", X.T, "\n")
        dW = (1 / num_Of_Input) * np.dot(X, dZ.T)  # each column are for each additional neuron in the layer
        print("dW: \n", dW, "\n")
        dB = (1 / num_Of_Input) * np.sum(dZ, axis=1, keepdims=True)
        # print("dB: \n", dB, "\n")

        W = W - step_Size * dW
        # print("W: \n", W, "\n")

        B = B - step_Size * dB
        # print("B: \n", B, "\n")


if __name__ == '__main__':
    main()
