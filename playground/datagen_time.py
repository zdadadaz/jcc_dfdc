#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 12:56:53 2020

@author: zdadadaz
"""
import keras
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from models.tcn import ED_TCN, BidirLSTM, Test_time_series

class DataGenerator_time(keras.utils.Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, dataframe, directory, x_col, y_col, target_size = (224,224),
                 to_fit=True, batch_size=32, dim=(256, 256),seq = 5,
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
        self.seq = seq
        self.target_size = target_size
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.rescale = rescale
        self.random_state = 100
        
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
        
        # print(indexes)
        files_batch = [[os.path.join(self.dataframe.iloc[k][4],self.dataframe.iloc[k][1],self.dataframe.iloc[k][0]), self.dataframe.iloc[k][5]] for k in indexes]
        y = []
        for k in indexes:
            if self.dataframe.iloc[k][1] == "REAL":
                y_tmp = np.ones((self.seq,1))
                # y_tmp = np.ones((self.seq-1,1))
                y.append(y_tmp)
            else:
                y_tmp = np.zeros((self.seq,1))
                # y_tmp = np.zeros((self.seq-1,1))
                y.append(y_tmp)
        y = np.array(y)
        # Generate data
        x = self.__data_generation(files_batch)
        # print( "x shape ????")
        # print(x.shape)
        # print( "y shape ????")
        # print(y.shape)
        # x = np.vstack(x)
        return np.array(x), y

       
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle == True:
            np.random.seed(100)
            np.random.shuffle(self.indexes)

    def __data_generation(self, files):
        imgs = []
        for img_file in files:
            image_path = img_file[0]
            number_frames = img_file[1]
            count = 0
            imgs_vid = []
            for i in range(0, number_frames, number_frames//self.seq):
                if count >= self.seq:
                    break
                count += 1
                path = os.path.join(self.directory, image_path) + "_" + str(i)+".txt"
                img = np.loadtxt(path, delimiter=',')
                ###############
                # Augment image
                ###############
                imgs_vid.append(img)
            # if count !=5:
            #     print("error")
            imgs_vid = np.stack(imgs_vid)
            imgs.append(imgs_vid)
        imgs = np.stack(imgs)
        # print(imgs.shape)
        return imgs

# path = './../../dataset/fb_db_xception/dfdc_train_part_0/REAL/aayrffkzxn.mp4_0.txt'
# x = np.loadtxt(path, delimiter=',')

# df = pd.read_csv('./dataset_vid_5.csv')
# df_train = df[df['split']=='test']
# train_generator=DataGenerator_time(dataframe = df_train,\
#                             directory = "./../../dataset/fb_db_xception/",\
#                             x_col="filename", y_col="label", \
#                             seq = 5,\
#                             target_size=(5,2048), \
#                             batch_size=10)
    
# n_feat = 2048
# max_len = 5
# conv = 25 #[25,20,5]
# causal = False
# n_nodes = [64, 96]
# n_classes = 1
# # ED_TCN
# model, param_str = ED_TCN(n_nodes, conv, n_classes, n_feat, max_len, causal=causal, 
#                                         activation='norm_relu', return_param_str=True) 
# # model, param_str = BidirLSTM(n_nodes[0], n_classes, n_feat, causal=causal, return_param_str=False)

# # print(model.summary())


# # def generator_test():
# #     while True:
# #         input_shape = (max_len,n_feat)
# #         test = np.random.random(input_shape)[np.newaxis,...]
# #         # np.array([1]), 
# #         yield np.array(test), 
# model.fit_generator(train_generator, steps_per_epoch=int(len(df_train)/1), epochs=10)