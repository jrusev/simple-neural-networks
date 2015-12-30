import theano
import theano.tensor as T
import numpy as np
import mnist

def init_weights(n_in, n_out):
    weights = np.random.randn(n_in, n_out) / np.sqrt(n_in)
    return theano.shared(np.asarray(weights, dtype=theano.config.floatX))

def feed_forward(X, w_h, w_o):
    h = T.nnet.sigmoid(T.dot(X, w_h))
    return T.nnet.softmax(T.dot(h, w_o))

trX, trY, teX, teY = mnist.load_data(one_hot=True)

w_h, w_o = init_weights(28*28, 100), init_weights(100, 10)
num_epochs, batch_size, learn_rate = 30, 10, 0.2

X, Y = T.fmatrices('X', 'Y')
y_ = feed_forward(X, w_h, w_o)

weights = [w_h, w_o]
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
