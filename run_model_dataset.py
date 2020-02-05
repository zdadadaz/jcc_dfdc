#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:29:49 2020

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
import cv2
from mtcnn import MTCNN
from keras.applications.xception import preprocess_input
from PIL import Image as pil_image

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

class Run_model():
    def __init__(self, name, dir_path, out_dir_path):
        self.name = name
        self.dir_path = dir_path
        self.out_dir_path = out_dir_path
        
    def run(self, classifier):
        tmp = ["dfdc_train_part_9", "dfdc_train_part_1", "dfdc_train_part_8","dfdc_train_part_23",
               "dfdc_train_part_19","dfdc_train_part_16","dfdc_train_part_29","dfdc_train_part_14"]

        # tmp = ["dfdc_train_part_9","dfdc_train_part_1","dfdc_train_part_8","dfdc_train_part_23",\
        #         "dfdc_train_part_19","dfdc_train_part_16", "dfdc_train_part_29"]
        #  to 28
        for folder in os.listdir(self.dir_path):
            if folder[0] =="." or folder in tmp:
                continue
            print(folder)
            for df_real in os.listdir(os.path.join(self.dir_path, folder)):
                if df_real[0] == ".":
                    continue
                for file in os.listdir(os.path.join(self.dir_path, folder,df_real)):
                    if  (file[-4:] != '.jpg'):
                        continue
                    outpath = os.path.join(self.out_dir_path, folder, df_real)              
                    # after run 29, remove 
                    if folder == "dfdc_train_part_13" and os.path.isfile(os.path.join(outpath,file[:-4]+".txt")):
                        continue
                    img_path = os.path.join(self.dir_path, folder,df_real, file)
                    
                    img = pil_image.open(img_path)
                    resample = 0
                    img = img.resize((299,299), resample)
                    img = np.expand_dims(img, axis=0)
                    img = preprocess_input(img)
                    # face = cv2.imread(img_path)
                    # inp = cv2.resize(face,(299,299))/255.
                    coef_img = classifier.predict(img)
                    self.write_coef(coef_img, outpath, file)
                    
    def write_coef(self, coef_img, path, file):
        np.savetxt(os.path.join(path,file[:-4]+".txt"), coef_img, delimiter=',')

dir_path = "./../dataset/fb_db"
out_dir_path = "./../dataset/fb_db_xception"
run = Run_model("xception", dir_path, out_dir_path)
classifier = Xception_main_noTop()
run.run(classifier)
                  
# make directories
# for i in range(50):
#     folder = "dfdc_train_part_"+str(i)
#     path = os.path.join(out_dir_path,folder)
#     cmd = 'mkdir ' + path
#     os.system(cmd)
    
#     folder = "dfdc_train_part_"+str(i)
#     path = os.path.join(out_dir_path,folder,"REAL")
#     cmd = 'mkdir ' + path
#     os.system(cmd)
    
#     folder = "dfdc_train_part_"+str(i)
#     path = os.path.join(out_dir_path,folder,"FAKE")
#     cmd = 'mkdir ' + path
#     os.system(cmd)
           
            