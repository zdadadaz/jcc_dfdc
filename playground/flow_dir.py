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
class DataGenerator_time(Sequence):
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

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        print(index)
        # Generate indexes of the batch
#        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
#        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        return [],0

        # Generate data
#        X = self._generate_X(list_IDs_temp)
#
#        if self.to_fit:
#            y = self._generate_y(list_IDs_temp)
#            return X, y
#        else:
#            return X
    
#    def check_para(self)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size,self.seq, *self.dim, self.n_channels))
        
        # Generate data
        for i in range(len(self.dataframe)):
            # Store sample
            vid = self.dataframe.iloc[i][0]
            file_path = os.path.join(self.directory,vid )
            if os.path.isfile(file_path):
                X[i,] = self._load_image(self.directory + self.labels[ID])
        return X

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((self.batch_size, *self.dim), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            y[i,] = self._load_grayscale_image(self.mask_path + self.labels[ID])

        return y

    def _load_image(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.rescale:
            img = img * self.rescale
        return img


df_train = pd.read_csv('training_dataset.csv')

#datagen=ImageDataGenerator(rescale=1./255)
#train_generator=datagen.flow_from_dataframe(dataframe=df_train, \
#                                            directory="./../db_multi_playground", \
#                                            x_col="filename", y_col="label", \
#                                            class_mode="binary", \
#                                            target_size=(32,32), \
#                                            batch_size=2,
#                                            shuffle=False)

train_generator=DataGenerator_time(dataframe = df_train,\
                           directory = "./../db_multi_playground",\
                           x_col="filename", y_col="label", \
                           target_size=(32,32), \
                           batch_size=1,\
                           rescale=1./255)
#train_generator=datagen.flow_from_dataframe(dataframe=df_train, \
#                                            directory="./../db_multi_playground", \
#                                            x_col="filename", y_col="label", \
#                                            class_mode="binary", \
#                                            target_size=(32,32), \
#                                            batch_size=2,
#                                            shuffle=False)

count = 0
for i in range(5):
    x,y=train_generator.next()
#    for j in range(len(x)):
#        image = x[j]
#        image *= 255
#        pyplot.subplot(330 + 1 + count)
#        count+= 1
#        pyplot.imshow(image.astype('uint8'))
       