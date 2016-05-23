import numpy as np
import mnist

def feed_forward(X, weights):
    a = [X]
    for w in weights:
        a.append(np.maximum(a[-1].dot(w),0))
    return a

def grads(X, Y, weights):
    grads = np.empty_like(weights)
    a = feed_forward(X, weights)
    delta = a[-1] - Y
    grads[-1] = a[-2].T.dot(delta)
    for i in xrange(len(a)-2, 0, -1):
        delta = (a[i] > 0) * delta.dot(weights[i].T)
        grads[i-1] = a[i-1].T.dot(delta)
    return grads / len(X)

trX, trY, teX, teY = mnist.load_data()
weights = [np.random.randn(*w) * 0.1 for w in [(784, 100), (100, 10)]]
num_epochs, batch_size, learn_rate = 30, 20, 0.1

for i in xrange(num_epochs):
    for j in xrange(0, len(trX), batch_size):
        X, Y = trX[j:j+batch_size], trY[j:j+batch_size]
        weights -= learn_rate * grads(X, Y, weights)
    prediction = np.argmax(feed_forward(teX, weights)[-1], axis=1)
    print i, np.mean(prediction == np.argmax(teY, axis=1))
