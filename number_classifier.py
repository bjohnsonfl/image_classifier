from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def logReg():
    in_size = 5  # length * width of image, but 1 for now
    num_Of_Input = 3  # number of images
    out_size = 2  # size of output
    step_Size = 0.1  # learning rate
    iter = 100  # iterations of gradient decent

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
        # print("dZ: \n", dZ, "\n")
        # print("X: \n", X, "\n")
        # print("XT: \n", X.T, "\n")
        dW = (1 / num_Of_Input) * np.dot(X, dZ.T)  # each column are for each additional neuron in the layer
        # print("dW: \n", dW, "\n")
        dB = (1 / num_Of_Input) * np.sum(dZ, axis=1, keepdims=True)
        # print("dB: \n", dB, "\n")

        W -= step_Size * dW
        # print("W: \n", W, "\n")

        B -= step_Size * dB
        # print("B: \n", B, "\n")


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
    step_Size = 0.1 # initial value in case you forget to set it

    # in_size are the number of neurons from prev layer, out_size is number of neurons in this layer
    def __init__(self, in_size, out_size, num_Of_Examples):
        self.num_Of_Input = num_Of_Examples
        self.W = np.random.randn(in_size, out_size)
        self.B = np.random.randn(out_size, 1)
        # A and dZ need to be returnable for feed forward and feedback
        self.A = np.zeros((out_size, num_Of_Examples))
        self.dZ = np.zeros((out_size, num_Of_Examples))
        self.num_Of_Examples = num_Of_Examples

    # use a property maybe? this function sets all instances with the same step size
    def setStepSize(self, size):
        self.step_Size = size

    def forward(self, X):
        # vectorized feedforward. W transpose weights * X training examples + B
        self.X = X
        Z = np.dot(self.W.T, self.X) + self.B
        self.A = 1 / (1 + np.exp(-Z))

        """ This is only for the last layer
        # Loss/Cost Function
        # i x j  matrix where i is number of outputs and j are examples
        J = -(Y * np.log(A) + ((1 - Y) * np.log(1 - A)))
        J = np.sum(J, axis=1, keepdims=True) / self.num_Of_Input
        # J = J.T # Make the output vertical
        """


    def feedback(self, W_Next, dZ_Next):
        # Gradient Decent Single Iteration
        # sigmoid derivative is g(z)(1-g(z)) where g(z) is the sigmoid function
        # dZ        dz1 = (W2 * dz2) .* g1'(z1) where g(z) is activation func
        # w2(szLay x szOut)  dz2 (out x numExam)

        self.dZ = np.dot(W_Next, dZ_Next) * np.multiply(self.A, (1 - self.A)) #not sure if this is correct

        dW = (1 / self.num_Of_Examples) * np.dot(self.X, self.dZ.T)  # each column are for each additional neuron in the layer
        dB = (1 / self.num_Of_Examples) * np.sum(self.dZ, axis=1, keepdims=True)

        self.W -= self.step_Size * dW
        self.B -= self.step_Size * dB

class OutputLayer(Layer):
    def __init__(self, in_size, out_size, num_Of_Examples, Y):
        super().__init__ (in_size, out_size, num_Of_Examples)
        self.Y = Y
        self.Jsum = np.zeros((out_size, 1))

    def forward(self, X):
        # vectorized feedforward. W transpose weights * X training examples + B
        self.X = X
        Z = np.dot(self.W.T, self.X) + self.B
        self.A = 1 / (1 + np.exp(-Z))

        # This is only for the last layer
        # Loss/Cost Function
        # i x j  matrix where i is number of outputs and j are examples
        J = -(self.Y * np.log(self.A) + ((1 - self.Y) * np.log(1 - self.A)))
        J = np.sum(J, axis=1, keepdims=True) / self.num_Of_Input
        self.Jsum = J

    def feedback(self):
        # Gradient Decent Single Iteration
        self.dZ = self.A - self.Y

        dW = (1 / self.num_Of_Examples) * np.dot(self.X, self.dZ.T)  # each column are for each additional neuron in the layer
        dB = (1 / self.num_Of_Examples) * np.sum(self.dZ, axis=1, keepdims=True)

        self.W -= self.step_Size * dW
        self.B -= self.step_Size * dB

    # property?
    def print_Cost(self):
        print(self.Jsum)


class Model:

    def __init__(self, layerModel, num_Of_Input):
        self.layerModel = layerModel
        self.num_Of_Input = num_Of_Input
        self.layers = []
        self.num_Of_Layers = 0


    """
    the layer model contains the size of each layer.
    The first number dictates the size of the input layer. 
    The last number is the output layer.
    If there are numbers remaining, those are the sizes of the hidden layers 
    [n0 out]
    [n0 n1 n2 out]
    """
    def generateLayers(self, Y):
        print(self.layerModel)
        self.num_Of_Layers = len(self.layerModel) - 1 # input layer does not count
        iter = 1

        # append each layer. the - 1 above excludes the output layer
        while iter != self.num_Of_Layers:
            layer = Layer(self.layerModel[iter - 1], self.layerModel[iter], self.num_Of_Input)
            self.layers.append(layer)
            iter += 1

        # add the output layer. Bad idea to require data Y while creating the model
        layer = OutputLayer(self.layerModel[iter - 1], self.layerModel[iter], self.num_Of_Input, Y)
        self.layers.append(layer)

    def feedForward(self, X):
        """
        layerTest10.forward(X)
        layerTest11.forward(layerTest10.A)
        layerTest12.forward(layerTest11.A)
        outputTest1.forward(layerTest12.A)
        """
        iter = 1
        self.layers[0].forward(X)
        while iter != self.num_Of_Layers:
            self.layers[iter].forward(self.layers[iter - 1].A)
            iter += 1

    def feedBack(self):
        """
        outputTest1.feedback()
        layerTest12.feedback(outputTest1.W, outputTest1.dZ)
        layerTest11.feedback(layerTest12.W, layerTest12.dZ)
        layerTest10.feedback(layerTest11.W, layerTest11.dZ)
        """
        iter = self.num_Of_Layers - 1
        self.layers[iter].feedback()

        iter -= 1
        while iter >= 0:
            self.layers[iter].feedback(self.layers[iter + 1].W, self.layers[iter + 1].dZ)
            iter -= 1

        self.layers[self.num_Of_Layers -1].print_Cost()



def main():
    #logReg()

    in_size = 32  # length * width of image, but 1 for now
    num_Of_Input = 1000  # number of images
    out_size = 4  # size of output
    step_Size = 0.1  # learning rate
    iter = 100  # iterations of gradient decent

    X = np.random.randint(0, 2, (in_size, num_Of_Input))  # input layer, tested with random int 0 to 1 for grayscale
    Y = np.random.rand(out_size, num_Of_Input)  # output_layer_truths where rows are outputs and col are examples


    layerTest10 = Layer(in_size, 10, num_Of_Input)
    layerTest11 = Layer(10, 10, num_Of_Input)
    layerTest12 = Layer(10, 10, num_Of_Input)
    outputTest1 = OutputLayer(10,out_size,num_Of_Input, Y)

    outputTest = OutputLayer(in_size, out_size, num_Of_Input, Y)
    outputTest.step_Size = 0.1

    layers = [in_size, 10, 10, out_size]
    model = Model(layers, num_Of_Input)
    model.generateLayers(Y)

    for _ in range(0, iter):

        # outputTest.forward(X)
        # outputTest.feedback()
        # outputTest.print_Cost()
        """
        layerTest10.forward(X)
        layerTest11.forward(layerTest10.A)
        layerTest12.forward(layerTest11.A)
        outputTest1.forward(layerTest12.A)

        outputTest1.feedback()
        layerTest12.feedback(outputTest1.W, outputTest1.dZ)
        layerTest11.feedback(layerTest12.W, layerTest12.dZ)
        layerTest10.feedback(layerTest11.W, layerTest11.dZ)


        outputTest1.print_Cost()
        """
        model.feedForward(X)
        model.feedBack()


if __name__ == '__main__':
    main()
