import numpy as np
import mnist

def feed_forward(X, weights):
    a = [X]
    for w in weights:
        a.append(sigmoid(a[-1].dot(w)))
    return a

def grads(X, Y, weights):
    grads = np.empty_like(weights)
    a = feed_forward(X, weights)
    delta = a[-1] - Y # cross-entropy
    grads[-1] = np.dot(a[-2].T, delta)
    for i in xrange(len(a)-2, 0, -1):
        delta = np.dot(delta, weights[i].T) * d_sigmoid(a[i])
        grads[i-1] = np.dot(a[i-1].T, delta)
    return grads / len(X)

sigmoid = lambda x: 1 / (1 + np.exp(-x))
d_sigmoid = lambda y: y * (1 - y)

trX, trY, teX, teY = mnist.load_data(one_hot=True)

weights = [
    np.random.randn(28*28, 100) / np.sqrt(28*28),
    np.random.randn(100, 10) / np.sqrt(100)]
num_epochs, batch_size, learn_rate = 30, 10, 0.2

for i in xrange(num_epochs):
    for j in xrange(0, len(trX), batch_size):
        X, Y = trX[j:j+batch_size], trY[j:j+batch_size]
        weights -= learn_rate * grads(X, Y, weights)
    prediction = np.argmax(feed_forward(teX, weights)[-1], axis=1)
    print i, np.mean(prediction == np.argmax(teY, axis=1))
