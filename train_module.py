#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:17:13 2020

@author: zdadadaz
"""

import numpy as np
from classifiers import *
from pipeline import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2
import random
import numpy as np

class Train():
    def __init__(self, name, classifier, batch_size=50, epochs = 20):
        self.name = name
        self.batch_size = batch_size
        self.epochs = epochs
        self.classifier = classifier
        self.history = None
        
    
    def augmentation(self):
        dataGenerator = ImageDataGenerator(rescale=1./255)
                                   # preprocessing_function = dft2 )
                                    # horizontal_flip=True,
                                    # # width_shift_range=50,
                                    # # height_shift_range=50,
                                    # zoom_range =0.2,
                                    # preprocessing_function = blur)
        val_dataGenerator = ImageDataGenerator(rescale=1./255)
                                               # preprocessing_function = dft2)
        return dataGenerator, val_dataGenerator
    
    def prepare_input(self):    
        dataGenerator, val_dataGenerator = self.augmentation()
        generator = dataGenerator.flow_from_directory(
                # 'deepfake_database/train_test',
                'db_small/train',
                target_size=(256, 256),
                batch_size=self.batch_size,
                class_mode='binary',
                subset='training')
        val_generator = val_dataGenerator.flow_from_directory(
                # 'deepfake_database/validation',
                'db_small/val',
                target_size=(256, 256),
                batch_size=self.batch_size,
                class_mode='binary')
        return generator, val_generator
    
    def callback_train(self):
        # checkpoint
        filepath="./weight_tmp/"+self.name +"-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        callbacks_list = [checkpoint]
        return callbacks_list
    
    def fit(self, tgen, vgen):
        history = self.classifier.fit_generator(tgen,int(tgen.samples/self.batch_size),self.epochs, \
                        self.callback_train(),vgen,int(vgen.samples/self.batch_size))
        self.plot_training_result(history)
        self.history = history
            
    def plot_training_result(self, history):
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model acc')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

class Train_fft(Train):
    def __init__(self, name, classifier, batch_size=50, epochs = 20):
        super().__init__(name, classifier, batch_size=50, epochs = 20)
    def augmentation(self):
        dataGenerator = ImageDataGenerator(preprocessing_function = self.dft2 )
        val_dataGenerator = ImageDataGenerator(preprocessing_function = self.dft2)
        return dataGenerator, val_dataGenerator
    
    def dft2(self, img):
        out = np.zeros(img.shape)
        def dft2_onechennel(image):
            f = np.fft.fft2(image)
            fshift = np.fft.fftshift(f)
            a = np.log(np.abs(fshift)+1e-9)
            a = a - a.mean()
            a = (a+1e-9) / (a.max()+1e-9)
            return a
        for i in range(3):
            out[:,:,i] = dft2_onechennel(img[:,:,i])
        return out
    
class Train_blur_compress(Train):
    def __init__(self, name, classifier, batch_size=50, epochs = 20):
        super().__init__(name, classifier, batch_size=50, epochs = 20)
    def augmentation(self):
        dataGenerator = ImageDataGenerator(rescale=1./255,
                                    preprocessing_function = self.blur_compress,
                                    horizontal_flip=True,
                                    brightness_range=[0.4,1.8])
        val_dataGenerator = ImageDataGenerator(rescale=1./255)
        return dataGenerator, val_dataGenerator
    
    def blur_compress(self, img):
        if random.random()<0.1:
            image = self.blur(img)
            return self.compress(image)
        else:
            return img
    
    def blur(self, img):
        sig = random.random()*3
        return (cv2.GaussianBlur(img,(5,5),sig)) 
    
    def compress(self, img):
        q = random.random()*70 + 30
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg
                        
    
name, classifier_fft, batch_size, epochs = "mesoInc4_fft", MesoInception4(), 50, 40
train_fft = Train_fft(name, classifier_fft, batch_size, epochs)
tgen, vgen = train_fft.prepare_input()
train_fft.fit(tgen, vgen)

name, classifier_aug, batch_size, epochs = "mesoInc4_aug", MesoInception4(), 50, 40
train_bc = Train_blur_compress(name, classifier_aug, batch_size, epochs)
tgen, vgen = train_bc.prepare_input()
train_bc.fit(tgen, vgen)
