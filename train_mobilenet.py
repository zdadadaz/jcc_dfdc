#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 20:58:00 2020

@author: zdadadaz
"""

from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenetv2 import MobileNetV2
from  keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop, SGD
import keras
from tensorflow import confusion_matrix
from matplotlib import pyplot as plt

import config
import numpy as np

train_path = 'data/train'
val_batch = 'data/val'
test_batch = 'data/test'

train_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(train_path, target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
                                                         class_mode='categorical', batch_size=20)
val_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(val_batch, target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
                                                         class_mode='categorical', batch_size=20)

def prepare_image(file):
    img = image.load_img(file, target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_expanded_dims)

mobilenet = MobileNetV2()

# x =  mobilenet.layers[-6].output
x =  mobilenet.layers[-2].output
predictions =  Dense(8, activation='softmax')(x)
from keras import Model
model = Model(inputs= mobilenet.input, outputs=predictions)

print(model.summary())


base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

# ==================== Toward data science website=========================================================
# x=base_model.output
# x=GlobalAveragePooling2D()(x)
# x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
# x=Dense(1024,activation='relu')(x) #dense layer 2
# x=Dense(512,activation='relu')(x) #dense layer 3
# preds=Dense(2,activation='softmax')(x) #final layer with softmax activation
# model=Model(inputs=base_model.input,outputs=preds)
# =============================================================================

# for layer in model.layers[:-5]:
#     layer.trainable = False


# for layer in model.layers[:-1]:
#     layer.trainable = False

print(model.summary())

# exit(0)


model.compile(SGD(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(train_batches, steps_per_epoch=10,
                    validation_data=val_batches, validation_steps=10, epochs=300, verbose=2)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# Get the ground truth from generator
ground_truth = train_batches.classes

# Get the label to class mapping from the generator
label2index = train_batches.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v, k) for k, v in label2index.items())

print(idx2label)


# _, val_labels =  next(val_batches)
#
# predictions = model.predict_generator(val_batches, steps=1, verbose=0)
#
# cm = confusion_matrix(val_batches, np.round(predictions[:,0]))
# cm_plot_labels = []
#
# for k, v in label2index.items():
#     cm_plot_labels.append(v)
#
# print(cm)



# serialize model to JSON
model_json = model.to_json()
with open("mobilenet.json", "w") as json_file:
    json_file.write(model_json)

from keras.models import save_model
save_model(model, 'mobilenet.h5')


import tensorflow as tf
# from tensorflow.contrib import lite
# tf.lite.TocoConverter

converter = tf.lite.TocoConverter.from_keras_model_file("mobilenet.h5")
tflite_model = converter.convert()
open("model/mobilenet.tflite", "wb").write(tflite_model)