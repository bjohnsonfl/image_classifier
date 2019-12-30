# data parser
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf


import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def parse_data(norm, normBias):

    # train_ds = tfds.load("mnist", split=tfds.Split.TRAIN, batch_size=-1)
    train_ds, test_ds = tfds.load('mnist:3.*.*', split=['train', 'test'], batch_size=-1)
    train = tfds.as_numpy(train_ds)
    test = tfds.as_numpy(test_ds)
    # numpy_images, numpy_labels = numpy_ds["image"], numpy_ds["label"]

    train_images, train_labels = train["image"] *norm + normBias, train["label"]
    test_images, test_labels = test["image"] *norm + normBias, test["label"]



    print("train_images: ", type(train_images), np.shape(train_images))
    print("train_labels: ", type(train_labels), np.shape(train_labels))
    print("test_images: ", type(test_images), np.shape(test_images))
    print("test_labels: ", type(test_labels), np.shape(test_labels))

    print(train_labels[0:5])

    train_images_stack = np.zeros((60000, 784))
    # train_labels_stack = np.reshape(train_labels, (60000, 1))
    train_labels_stack = np.zeros((60000, 10))

    test_images_stack = np.zeros((10000, 784))
    # test_labels_stack = np.reshape(test_labels, (10000, 1))
    test_labels_stack = np.zeros((10000, 10))

    print("train_images_stack: ", type(train_images_stack), np.shape(train_images_stack))
    print("train_labels_stack: ", type(train_labels_stack), np.shape(train_labels_stack))
    print("test_images_stack: ", type(test_images_stack), np.shape(test_images_stack))
    print("test_labels_stack: ", type(test_labels_stack), np.shape(test_labels_stack))



    # creates a 60000 x 784 matrix for train images
    for i in range (0, 60000):
        list = []
        for j in range (0, 28):
            horz = train_images[i, j, :, 0]
            horz = np.reshape(horz, (1,28))
            list.append(horz)
        train_images_stack[i, :] = np.hstack(list)



    # creates a 10000 x 784 matrix for test images

    for i in range(0, 10000):
        list = []
        for j in range(0, 28):
            horz = test_images[i, j, :, 0]
            horz = np.reshape(horz, (1, 28))
            list.append(horz)
        test_images_stack[i, :] = np.hstack(list)


    """ 
    debug
    var = np.zeros((5, 784))
    var[0:5, :] = np.reshape(test_images_stack[0, :], (1, 784))
    im = plt.imshow(var)
    plt.show()
    """

    # creates a 60000 x 10 matrix for test labels
    for i in range(0, 60000):
        # given a number num, 0 - 9, place a 1 at index num in an array of size 10
        # i.e. if num = 5, [0 0 0 0 0 1 0 0 0 0]
        num = train_labels[i]
        train_labels_stack[i, num] = 1


    # creates a 10000 x 10 matrix for test labels
    for i in range(0, 10000):
        # given a number num, 0 - 9, place a 1 at index num in an array of size 10
        # i.e. if num = 5, [0 0 0 0 0 1 0 0 0 0]
        num = test_labels[i]
        test_labels_stack[i, num] = 1

    return train_images_stack.T, train_labels_stack.T, test_images_stack.T, test_labels_stack.T


