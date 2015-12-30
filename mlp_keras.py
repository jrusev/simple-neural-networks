import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

trX, trY, teX, teY = mnist.load_data(one_hot=True)

net = Sequential([
    Dense(100, input_dim=28*28, activation='sigmoid'),
    Dense(10, activation='softmax')
])

num_epochs, batch_size, learn_rate = 30, 10, 0.2

net.compile(SGD(learn_rate), 'categorical_crossentropy', metrics=['accuracy'])
net.fit(trX, trY, batch_size, num_epochs, verbose=1, validation_data=(teX, teY))
