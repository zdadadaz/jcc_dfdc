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
from sklearn.utils import resample
import cv2
from sklearn.metrics import log_loss
from mtcnn import MTCNN


class Run_model():
    def __init__(self, name, dir_path):
        self.name = name
        self.dir_path = dir_path
        
    def run(self):
        for i in range(len(self.df_train)):
            