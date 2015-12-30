import tensorflow as tf
import mnist
from math import sqrt

def init_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=1 / sqrt(shape[0])))

def feed_forward(X, w1, w2, w3):
    l1_conv = tf.nn.sigmoid(tf.nn.conv2d(X, w1, strides=[1, 1, 1, 1], padding='VALID'))
    l1_pool = tf.nn.max_pool(l1_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    l1_pool_flat = tf.reshape(l1_pool, [-1, w2.get_shape().as_list()[0]])
    l2 = tf.nn.sigmoid(tf.matmul(l1_pool_flat, w2))
    return tf.matmul(l2, w3)

trX, trY, teX, teY = mnist.load_data(one_hot=True, reshape=(-1, 28, 28, 1))

# conv8(5x5) -> pool(2x2) -> dense100 -> softmax10
w1 = init_weights([5, 5, 1, 8])
w2 = init_weights([8*12*12, 100])
w3 = init_weights([100, 10])

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

y_ = feed_forward(X, w1, w2, w3)

num_epochs, batch_size, learn_rate = 30, 10, 0.2
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, Y))
train = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,1), tf.argmax(Y,1)), tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(num_epochs):
    for j in xrange(0, len(trX), batch_size):
        batch_x, batch_y = trX[j:j+batch_size], trY[j:j+batch_size]
        sess.run(train, feed_dict={X: batch_x, Y: batch_y})
    print i, sess.run(accuracy, feed_dict={X: teX, Y: teY})
