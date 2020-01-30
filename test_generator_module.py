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
from keras.applications.xception import preprocess_input
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
        self.final_score = None
    def augmentation(self):
        dataGenerator = ImageDataGenerator(preprocessing_function=preprocess_input)
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
        
    def selftesting(self):
        from PIL import Image as pil_image
        
        scores = []
        gt = []
        epoch_step = int(len(self.df_train)/self.batch_size)
        for i in range(epoch_step):
            ss = i * self.batch_size
            indexes = [j for j in range(ss,ss+self.batch_size)]
            images = []
            print("epoch " + str(i) + " / " + str(epoch_step))
            for j in indexes:
                img_path = os.path.join("./../dataset/fb_db", self.df_train.iloc[j,0])
                # imgqq = pil_image.open(img_path)
                
                img1 = cv2.imread(img_path)
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                # img = pil_image.fromarray(img1,mode="RGB")
                # resample = 0
                # img = img.resize((299,299), resample)
                
                img = cv2.resize(img1,(299,299), interpolation=cv2.INTER_NEAREST)
                images.append(img)
                gt.append(list(self.df_train.iloc[j,1])[0])
            images = np.stack(images)
            images = preprocess_input(images)
            re_imgs = classifier.predict(np.array(images))
            for j in re_imgs:
                scores.append(1.-j[0])
                
        # self.final_score = scores
        submit_score = [[i, 1-i] for i in scores]
        tmp_final_score = log_loss(gt, submit_score)
        self.final_score = tmp_final_score
        print(self.name+" final_score: " + str(tmp_final_score))
        

name, classifier, batch_size = "xception", Xception_main(), 10
valid_path = "./playground/test_dataset_5.csv"
classifier.load('./result/xception/x1.1.1/xception-02-0.34.hdf5')
# classifier.load("result/xception/x1.0.0/xception-08-0.66.hdf5")
evaluation = Test_gen(name, classifier,valid_path, batch_size=batch_size)
tgen = evaluation.prepare_input()

evaluation.selftesting()
# evaluation.evaluate(tgen)