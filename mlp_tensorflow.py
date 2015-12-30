import tensorflow as tf
import numpy as np
import mnist

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def feed_forward(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)

trX, trY, teX, teY = mnist.load_data(one_hot=True)

w_h, w_o = init_weights([28*28, 100]), init_weights([100, 10])
num_epochs, batch_size, learn_rate = 30, 10, 0.2

X = tf.placeholder("float", [None, 28*28])
Y = tf.placeholder("float", [None, 10])
y_ = feed_forward(X, w_h, w_o)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, Y))
train = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(num_epochs):
    for j in xrange(0, len(trX), batch_size):
        batch_x, batch_y = trX[j:j+batch_size], trY[j:j+batch_size]
        sess.run(train, feed_dict={X: batch_x, Y: batch_y})
    prediction = sess.run(tf.argmax(y_, 1), feed_dict={X: teX, Y: teY})
    print i, np.mean(prediction == np.argmax(teY, axis=1))
