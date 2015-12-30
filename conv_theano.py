import theano
import theano.tensor as T
import numpy as np
import mnist
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d

def init_weights(shape):
    weights = np.random.randn(*shape) / np.sqrt(shape[0])
    return theano.shared(np.asarray(weights, dtype=theano.config.floatX))

def feed_forward(X, weights, pool_size=(2, 2)):
    l1_conv = T.nnet.sigmoid(conv2d(X, weights[0]))
    l1_pool = pool_2d(l1_conv, pool_size, ignore_border=True)
    l2 = T.nnet.sigmoid(T.dot(l1_pool.flatten(ndim=2), weights[1]))
    return T.nnet.softmax(T.dot(l2, weights[2]))

trX, trY, teX, teY = mnist.load_data(one_hot=True, reshape=(-1, 1, 28, 28))

# conv8(5x5) -> pool(2x2) -> dense100 -> softmax10
shapes = (8, 1, 5, 5), (8*12*12, 100), (100, 10)
weights = [init_weights(shape) for shape in shapes]
X, Y = T.ftensor4(), T.fmatrix()
y_ = feed_forward(X, weights)

num_epochs, batch_size, learn_rate = 30, 10, 0.2

grads = T.grad(cost=T.nnet.categorical_crossentropy(y_, Y).mean(), wrt=weights)
train = theano.function(
    inputs=[X, Y],
    updates=[[w, w - g * learn_rate] for w, g in zip(weights, grads)],
    allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=T.argmax(y_, axis=1))

for i in range(num_epochs):
    for j in xrange(0, len(trX), batch_size):
        train(trX[j:j+batch_size], trY[j:j+batch_size])
    print i, np.mean(predict(teX) == np.argmax(teY, axis=1))
