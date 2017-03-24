# coding: utf-8

import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np

def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        print "loading MNIST training files"
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        print "loading MNIST testing files"
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

if __name__ == "__main__"
    im, la = load_mnist()
    print "Writing training files"
    im = [i.flatten() for i in im]
    np.savetxt('train_vectors', im, delimiter=',', fmt='%d')
    np.savetxt('train_labels', la, fmt='%d')


    im2, la2 = load_mnist(dataset="testing")

    print "Writing testing files"
    im2 = [i.flatten() for i in im2]
    np.savetxt('test_vectors', im2, delimiter=',', fmt='%d')
    np.savetxt('test_labels', la2, fmt='%d')



