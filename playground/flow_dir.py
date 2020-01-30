#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:37:42 2020

@author: liulara
"""
import keras
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence

#train_generator=DataGenerator_time(dataframe = df_train,\
#                           directory = "./../db_multi_playground",\
#                           x_col="filename", y_col="label", \
#                           target_size=(32,32), \
#                           batch_size=2,\
#                           rescale=1./255)

#Reference
#https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c
#https://stackoverflow.com/questions/43086548/how-to-manually-specify-class-labels-in-keras-flow-from-directory

#yeild
#https://stackoverflow.com/questions/39325275/how-to-train-tensorflow-network-using-a-generator-to-produce-inputs
#https://stackoverflow.com/questions/56079223/custom-keras-data-generator-with-yield
#https://medium.com/@fromtheast/implement-fit-generator-in-keras-61aa2786ce98
#https://keras.io/models/sequential/
# https://stackoverflow.com/questions/54590826/generator-typeerror-generator-object-is-not-an-iterator/57101352
# https://stackoverflow.com/questions/55889389/keras-utils-sequence-object-is-not-an-iterator
# class DataGenerator_time(Sequence):
class DataGenerator_time(keras.utils.Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, dataframe, directory, x_col, y_col, target_size = (224,224),
                 to_fit=True, batch_size=32, dim=(256, 256),seq = 1,
                 n_channels=3, n_classes=1, shuffle=False, rescale =None):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.dataframe = dataframe.copy()
        self.directory = directory
        self.x_col = x_col
        self.y_col = y_col
        self.target_size = target_size
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.rescale = rescale
        
    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.dataframe) / self.batch_size))
        # return 10

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        print(indexes)
        files_batch = [self.dataframe.iloc[k][0] for k in indexes]
        y = []
        for k in indexes:
            if self.dataframe.iloc[k][1] == "REAL":
                y.append(1)
            else:
                y.append(0)
        
        # Generate data
        x = self.__data_generation(files_batch)
        print(x.shape)
        # x = np.vstack(x)
        return x, y

       
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle == True:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)

    def __data_generation(self, files):
        imgs = []

        for img_file in files:
            path = os.path.join(self.directory, img_file)
            # print(path)
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(self.target_size[0],self.target_size[1]))
            ###############
            # Augment image
            ###############
            imgs.append(img)
        imgs = np.stack(imgs)
        return imgs


df_train = pd.read_csv('./training_dataset_5.csv')

#datagen=ImageDataGenerator(rescale=1./255)
#train_generator=datagen.flow_from_dataframe(dataframe=df_train, \
#                                            directory="./../db_multi_playground", \
#                                            x_col="filename", y_col="label", \
#                                            class_mode="binary", \
#                                            target_size=(32,32), \
#                                            batch_size=2,
#                                            shuffle=False)

train_generator=DataGenerator_time(dataframe = df_train,\
                           directory = "./../../dataset/fb_db/",\
                           x_col="filename", y_col="label", \
                           target_size=(32,32), \
                           batch_size=10)
#train_generator=datagen.flow_from_dataframe(dataframe=df_train, \
#                                            directory="./../db_multi_playground", \
#                                            x_col="filename", y_col="label", \
#                                            class_mode="binary", \
#                                            target_size=(32,32), \
#                                            batch_size=2,
#                                            shuffle=False)

# count = 0
# for i in range(5):
    # x,y=train_generator.next()
#    for j in range(len(x)):
#        image = x[j]
#        image *= 255
#        pyplot.subplot(330 + 1 + count)
#        count+= 1
#        pyplot.imshow(image.astype('uint8'))
       
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


x = Input(shape = (32, 32, 3))
# x2 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x)
# x2 = BatchNormalization()(x2)
# x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
# # y = Flatten()(x2)
# x3 = Reshape((-1, 16))(x2)
# y = SimpleRNN(hidden_units,
#             kernel_initializer=initializers.RandomNormal(stddev=0.001),
#             recurrent_initializer=initializers.Identity(gain=1.0),
#             activation='relu', input_shape=(3,128*128,16))(x3)

y = Dense(1)(x)

# model.add(Dense(1))
model = KerasModel(inputs = x, outputs = y)
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='binary_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])
print(model.summary)
model.fit_generator(train_generator, steps_per_epoch=10, epochs=1)

