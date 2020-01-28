#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 09:03:18 2020

@author: zdadadaz
"""
import numpy as np
from classifiers import *
from pipeline import *
import os
import imageio
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import json
import pandas as pd
from sklearn.utils import resample
import cv2
from sklearn.metrics import log_loss
from keras.preprocessing.image import ImageDataGenerator

# import seaborn as sns
# import matplotlib.pylab as plt


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

class Test_gen():
    def __init__(self, name, classifier, train_path, batch_size = 1):
        self.name = name
        self.batch_size = batch_size
        self.classifier = classifier
        self.df_train = pd.read_csv(train_path)
        self.accuracy = None
        self.loss = None
    def augmentation(self):
        dataGenerator = ImageDataGenerator(rescale=1./255)
        return dataGenerator
    
    def prepare_input(self):    
        dataGenerator = self.augmentation()
        
        generator = dataGenerator.flow_from_dataframe(dataframe=self.df_train, \
                                            directory="./../dataset/fb_db", \
                                            x_col="filename", y_col="label", \
                                            class_mode="binary", \
                                            target_size=(299,299), \
                                            batch_size=self.batch_size,\
                                            shuffle=False)
        return generator
    

    def evaluate(self, tgen):
        print(self.name + " is evaluating ... ")
        loss,accuracy = self.classifier.eval_generator( tgen, int(tgen.samples/self.batch_size))
        print(self.name+" accuracy: " + str(accuracy))
        print(self.name+" loss: " + str(loss))
        self.accuracy = accuracy
        self.loss = loss
        

name, classifier, batch_size = "xception", Xception_main(), 10
valid_path = "./playground/test_dataset_5.csv"
classifier.load("weight_tmp/xception-02-0.39.hdf5")
# classifier.load("result/xception/x1.0.0/xception-08-0.66.hdf5")
evaluation = Test_gen(name, classifier,valid_path, batch_size=batch_size)
tgen = evaluation.prepare_input()
evaluation.evaluate(tgen)