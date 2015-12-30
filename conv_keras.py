import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

trX, trY, teX, teY = mnist.load_data(one_hot=True, reshape=(-1, 1, 28, 28))

model = Sequential()
model.add(Convolution2D(8, 5, 5, input_shape=trX.shape[1:]))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

num_epochs, batch_size, learn_rate = 30, 10, 0.2

model.compile(SGD(learn_rate), 'categorical_crossentropy', metrics=['accuracy'])
model.fit(trX, trY, batch_size, num_epochs, verbose=1, validation_data=(teX, teY))
