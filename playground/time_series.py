#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 20:20:27 2020

@author: liulara
"""
# Reference 
# https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from numpy import array
import keras
k_init = keras.initializers.Constant(value=0.1)
b_init = keras.initializers.Constant(value=0)
r_init = keras.initializers.Constant(value=0.1)
# LSTM units
units = 1

# define model
inputs1 = Input(shape=(3, 2))
lstm1 = LSTM(units, return_sequences=True, kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init)(inputs1)
model = Model(inputs=inputs1, outputs=lstm1)
# define input data
data = array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3]).reshape((1,3,2))
# make and show prediction
output = model.predict(data)
print(output, output.shape)



#name, classifier, batch_size, epochs = "xception_lr", Meso_lstm(), 1, 1
# check output of model
#model = classifier.init_model()
#outputs = [layer.output for layer in model.layers[1:]]   
#print(outputs)
#input_shape = (32, 5, 256,256,3)
#data = np.random.random(input_shape)
## make and show prediction
#out = model.predict(data)
#print(model.summary())