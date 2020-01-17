#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 15:24:56 2020

@author: zdadadaz
"""


import numpy as np
from classifiers import *
from pipeline import *
import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

classifier = MesoInception4()
classifier.load('model_bs_ep20.h5')

dirname = "./../fb_whole/dfdc_train_part_21"
# dirname = "test_videos/"
# dirname = "/kaggle/input/deepfake-detection-challenge/test_videos/"

submit = []
videos = os.listdir(dirname)
for v in range(400):
    vi = videos[v]
    re_video = 0.5
    try:
        frame_subsample_count = 30
        # Compute face locations and store them in the face finder
        face_finder = FaceFinder(join(dirname, vi), load_first_face = False)
        skipstep = max(floor(face_finder.length / frame_subsample_count), 0)
        face_finder.find_faces(resize=0.5, skipstep = skipstep)
        re_video, p = compute_onevideo_accuracy(classifier, face_finder)    
        if np.isnan(re_video):
            re_video = 0.5
    except:
        re_video = 0.5
    submit.append([vi,1.0-re_video])