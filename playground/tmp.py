# -*- coding: utf-8 -*-
from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Conv2D
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Concatenate, LeakyReLU, GlobalMaxPooling2D, Reshape

from keras.layers import SimpleRNN
from keras import initializers
from keras.optimizers import RMSprop
from keras import backend as K
import numpy as np
from keras.models import Model as KerasModel

batch_size = 32
num_classes = 10
epochs = 2
# hidden_units = 100
hidden_units = 10

learning_rate = 1e-6
clip_norm = 1.0

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1, 1)
x_test = x_test.reshape(x_test.shape[0], -1, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

input_shape = (256,256,3)

print('Evaluate IRNN...')
# model = Sequential()
# model.add(SimpleRNN(hidden_units,
#                     kernel_initializer=initializers.RandomNormal(stddev=0.001),
#                     recurrent_initializer=initializers.Identity(gain=1.0),
#                     activation='relu',
#                     input_shape=x_train.shape[1:]))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))

x = Input(shape = (256, 256, 3))
x2 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x)
x2 = BatchNormalization()(x2)
x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
# y = Flatten()(x2)
x3 = Reshape((-1, 16))(x2)
y = SimpleRNN(hidden_units,
            kernel_initializer=initializers.RandomNormal(stddev=0.001),
            recurrent_initializer=initializers.Identity(gain=1.0),
            activation='relu', input_shape=(3,128*128,16))(x3)

# model.add(SimpleRNN(hidden_units,
#             kernel_initializer=initializers.RandomNormal(stddev=0.001),
#             recurrent_initializer=initializers.Identity(gain=1.0),
#             activation='relu'))
            # input_shape=x_train.shape[1:]))
# model.add(Dense(1))
model = KerasModel(inputs = x, outputs = y)
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])


inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers[1:]]          # all layer outputs
functors = [K.function([inp], [out]) for out in outputs] 

test = np.random.random(input_shape)[np.newaxis,...]
layer_outs = [func([test]) for func in functors]
# print (layer_outs)
print(outputs[-1])
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))

# scores = model.evaluate(x_test, y_test, verbose=0)
# print('IRNN test score:', scores[0])
# print('IRNN test accuracy:', scores[1])