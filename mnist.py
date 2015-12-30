import os
import gzip
import numpy as np

DATA_URL = 'http://yann.lecun.com/exdb/mnist/'

def load_data(one_hot=True, reshape=None, validation_size=10000):
    """Download and import the MNIST dataset from Yann LeCun's website.
    Reserve 10,000 examples from the training set for validation.

    Params:
        one_hot: boolean, whether to convert the labels to one-hot vectors.
        reshape: a list of ints, to optionally reshape the images.
        validation_size: int, reserve this number of examples for validation.

    Returns:
        x_tr: training images - numpy array of shape (50000, 784). Each row
            represents the pixels of an image as numpy array of 784 (28 x 28)
            float values between 0 and 1 (0 stands for black, 1 for white).
        y_tr: training labels - a numpy array of 50000 entries. Each entry is
            a number from 0 to 9 indicating which digit the image represents.
        x_te: testing images - numpy array of shape (10000, 784).
        y_te: testing labels - a numpy array of 10000 entries.
    """
    x_tr = load_images('train-images-idx3-ubyte.gz')
    y_tr = load_labels('train-labels-idx1-ubyte.gz')
    x_te = load_images('t10k-images-idx3-ubyte.gz')
    y_te = load_labels('t10k-labels-idx1-ubyte.gz')

    # Rserve the last `validation_size` training examples for validation.
    x_tr = x_tr[:-validation_size]
    y_tr = y_tr[:-validation_size]

    if one_hot:
        y_tr, y_te = [to_one_hot(y) for y in (y_tr, y_te)]

    if reshape:
        x_tr, x_te = [x.reshape(*reshape) for x in (x_tr, x_te)]

    return x_tr, y_tr, x_te, y_te

def load_images(filename):
    maybe_download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # The data is just a long 1-dim array, so we need to shape it into a matrix.
    data = data.reshape(-1, 28 * 28)
    # The inputs come as bytes, we convert them to float32 in range [0,1].
    # (Actually to range [0, 255/256], for compatibility to the version
    # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
    return data / np.float32(256)

def load_labels(filename):
    maybe_download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def maybe_download(filename):
    """Download the file, unless it's already here."""
    if not os.path.exists(filename):
        from urllib import urlretrieve
        print("Downloading %s" % filename)
        urlretrieve(DATA_URL + filename, filename)

def to_one_hot(labels, num_classes=10):
    """"Convert class labels from scalars to one-hot vectors."""
    return np.eye(num_classes)[labels]
