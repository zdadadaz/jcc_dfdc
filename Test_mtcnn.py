#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:21:51 2020

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
import matplotlib.pyplot as plt
import seaborn as sns

def clip(value, shift = 0.1):
    if value < shift:
        return shift
    if value >(1-shift):
        return (1-shift)
    return value

def detect_mtcnn(detector, images, margin = 0):
    faces = []
    count = 1
    for image in images:
        boxes = detector.detect_faces(image)
        if len(boxes) == 0:
            continue
        box = boxes[0]['box']
        face_position = [0]*4
        # tmp[0] = box[1]
        # tmp[2] = box[1]+box[3]
        # tmp[3] = box[0]
        # tmp[1] = box[0]+box[2]
        maxframe = max(int(box[3]/2),int(box[2]/2))
        center_y = box[1]+int(box[3]/2)
        center_x = box[0]+int(box[2]/2)
        face_position[0] = center_y-maxframe
        face_position[2] = center_y+maxframe
        face_position[3] = center_x-maxframe
        face_position[1] = center_x+maxframe
        offset = round(margin * (face_position[2] - face_position[0]))
        y0 = max(face_position[0] - offset, 0)
        x1 = min(face_position[1] + offset, image.shape[1])
        y1 = min(face_position[2] + offset, image.shape[0])
        x0 = max(face_position[3] - offset, 0)
        face = image[y0:y1,x0:x1]
        
        # face = image[box[1]:box[3]+box[1], box[0]:box[2]+box[0]]
        if sum(np.array(face.shape)==0) == 1:
            continue
        # imgplot = plt.imshow(face)
        # plt.show()
        face = cv2.resize(face,(256,256))/255.
        # score = classifier.predict(np.array([inp]))
        # print("score: " + str(score))
        faces.append(face)
    return faces


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

classifier = MesoInception4()
classifier.load('weight_tmp/meso4inc-01-0.69.hdf5')
# classifier.load('result/meso4inc_base/v1.0.2/meso4inc-04-0.74.hdf5')

dirname = "./../fb_whole/dfdc_train_part_21"
# dirname = "test_videos/"
# dirname = "/kaggle/input/deepfake-detection-challenge/test_videos/"

detector = MTCNN()

dir_json = './../fb_whole/metadata_21.json'

files =[]
with open(dir_json) as json_file:
    data = json.load(json_file)
    files = pd.DataFrame.from_dict(data, orient='index')
    files.reset_index(level=0, inplace=True)
real = resample(files[files['label']=='REAL'], n_samples=200, replace=False, random_state=800)
fake = resample(files[files['label']=='FAKE'], n_samples=200, replace=False, random_state=100)
total_tests = real.append(fake)
total_tests.reset_index(level=0, inplace=True,drop=True)

submit = []

save_interval = 150 # perform face detection every {save_interval} frames
margin = 0.2
# for vi in os.listdir(dirname):
videos = os.listdir(dirname)
for v in range(400):
    # vi = videos[v]
    print("index: "+str(v)+" / 400")
    vi = total_tests.iloc[v][0]
#     print(os.path.join("/kaggle/input/deepfake-detection-challenge/test_videos/", vi))
    re_video = 0.5
    try:
        # read video
        reader = cv2.VideoCapture(os.path.join(dirname, vi))
        images = []
        for i in range(int(reader.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, image = reader.read()
            if i % save_interval != 0:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        reader.release()
        images = np.stack(images)
        re_imgs = []
        # detect face 
        faces = detect_mtcnn(detector, images)
        # for i in range(len(faces)):
        #     imageio.imwrite(join('./tmp_mtcnn',vi+"_"+str(i)+"_"+str(0)+".jpg"),faces[i],'jpg')
        re_imgs = classifier.predict(np.array(faces))
        re_video = np.average(re_imgs)
        # re_video = clip(re_video)
        if np.isnan(re_video):
            re_video = 0.5
        print("score = "+str(re_video))
    except:
        re_video = 0.5
        print("score = "+str(re_video))
    submit.append([vi,1.0-re_video])

submit_score = [[i[1], 1-i[1]] for i in submit]
final_score = log_loss(list(total_tests['label']), submit_score)

submission = pd.DataFrame(submit, columns=['filename', 'label']).fillna(0.5)
submission.sort_values('filename').to_csv('submission_mtcn.csv', index=False)
# submission.to_csv('submission_mtcn.csv', index=False)
# sns.distplot(list(submission['label']));

submit_score_clip = [[clip(i[1]), 1-clip(i[1])] for i in submit]
final_score_clip = log_loss(list(total_tests['label']), submit_score_clip)

print("mtcnn final_score = " + str(final_score))
print("mtcnn final_score clip = " + str(final_score_clip))