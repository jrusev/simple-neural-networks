# Simple Neural Networks

Introductory examples in deep learning may sometimes be too verbose, often up to
300 lines, which makes it hard to actually see what is going on. This repo is an
attempt to fix this - the longest example is 39 lines (31 LOC). It contains the
same neural network implemented with 5 different libraries -
[Numpy](http://www.numpy.org/), [Theano](http://www.deeplearning.net/software/theano/),
[TensorFlow](https://www.tensorflow.org/), [Keras](http://keras.io/) and
[Torch](http://torch.ch/). The network is a simple multilayer perceptron with a
hidden layer of 100 neurons and an output layer with 10 neurons, and is trained
on the [MNIST](http://yann.lecun.com/exdb/mnist/) database of handwritten digits.
It can achieve accuracy of 97.8%.

```shell
$ python mlp_numpy.py
$ python mlp_theano.py
$ python mlp_tensorflow.py
$ python mlp_keras.py
$ th mlp_torch.lua
```

A more detailed explanation of the implementation with Numpy can be found in my blog
post "[Hacking MNIST in 30 Lines of Python](http://jrusev.github.io/post/hacking-mnist/)".

## MNIST dataset

The MNIST dataset consists of handwritten digit images and is divided in 60,000
examples for the training set and 10,000 examples for testing. We use a small
[script](./mnist.py) to download the MNIST data and load it to memory. By default
it reserves 10,000 examples from the official training set for validation,
so all neural nets train with 50,000 examples.

## Convolutional Neural Network

For completeness, I also included a conv net trained on MNIST (implemented with
Theano, TensorFlow and Keras). The last two layers are the same as in the MLP,
but now there is a convolutional layer in front (8 kernels of size 5x5, with 2x2
max pooling). This improves the accuracy to 98.7%. Run with:

```shell
$ THEANO_FLAGS='floatX=float32' python conv_theano.py
$ python conv_tensorflow.py
$ python conv_keras.py
```

You can reach 99.0% accuracy (99.1% using Keras) with the following architecture:

```
conv8(5x5) -> conv16(5x5) -> pool2 -> fc100 -> softmax10
```
