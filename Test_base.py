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

import seaborn as sns
import matplotlib.pylab as plt



def detect_mtcnn(detector, image):
    boxes = detector.detect_faces(image)
    if len(boxes) == 0:
        return 0.5
    box = boxes[0]['box']
    face = image[box[1]:box[3]+box[1], box[0]:box[2]+box[0]]
    inp = cv2.resize(face,(256,256))/255.
    return inp

def clip(value, shift = 0.1):
    if value < shift:
        return shift
    if value >(1-shift):
        return (1-shift)
    return value

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

classifier = MesoInception4()
# classifier.load('model_bs_ep20.h5')
classifier.load('weight_tmp/meso4inc-20-0.81.hdf5')
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
    # if vi != "aopnftmaex.mp4":
    #     continue
#     print(os.path.join("/kaggle/input/deepfake-detection-challenge/test_videos/", vi))
    re_video = 0.5
    try:
        video = Video(os.path.join(dirname, vi))
        re_imgs = []
        for i in range(0,video.__len__(),save_interval):
            img = video.get(i)
            face_positions = face_recognition.face_locations(img)
            
            # debug
            # boxes = detector.detect_faces(img)
            # box = boxes[0]['box']
            # tmp = [0]*4
            # tmp[0] = box[1]+int(box[3]/2)-128
            # tmp[2] = box[1]+int(box[3]/2)+128
            # tmp[3] = box[0]+int(box[2]/2)-128
            # tmp[1] = box[0]+int(box[2]/2)+128
            # maxframe = max(int(box[3]/2),int(box[2]/2))
            # center_y = box[1]+int(box[3]/2)
            # center_x = box[0]+int(box[2]/2)
            # tmp[0] = center_y-maxframe
            # tmp[2] = center_y+maxframe
            # tmp[3] = center_x-maxframe
            # tmp[1] = center_x+maxframe
            # ff = face_positions[0]
            # face1 = img[ff[0]:ff[2],ff[3]:ff[1]]
            # face2 = img[tmp[0]:tmp[2],tmp[3]:tmp[1]]
            # imgplot = plt.imshow(face1)
            # plt.show()
            # imgplot1 = plt.imshow(face2)
            # plt.show()
            
            count = 0
            for face_position in face_positions:
                offset = round(margin * (face_position[2] - face_position[0]))
                y0 = max(face_position[0] - offset, 0)
                x1 = min(face_position[1] + offset, img.shape[1])
                y1 = min(face_position[2] + offset, img.shape[0])
                x0 = max(face_position[3] - offset, 0)
                face = img[y0:y1,x0:x1]
                inp = cv2.resize(face,(256,256))/255.
                # imageio.imwrite(join('./tmp_base',vi+"_"+str(i)+"_"+str(count)+".jpg"),inp,'jpg')
                count+=1
                re_img = classifier.predict(np.array([inp]))
    #             print(vi,": ",i , "  :  ",classifier.predict(np.array([inp])))
                re_imgs.append(re_img[0][0])
        re_video = np.average(re_imgs)
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
submission.sort_values('filename').to_csv('submission_base.csv', index=False)
# submission.to_csv('submission_base.csv', index=False)
# sns.distplot(list(submission['label']));

submit_score_clip = [[clip(i[1]), 1-clip(i[1])] for i in submit]
final_score_clip = log_loss(list(total_tests['label']), submit_score_clip)

# submit_score_clip_tmp = [ [0.6, 0.4] if i[1] == 0.5 else [clip(i[1]), 1-clip(i[1])] for i in submit]
# final_score_clip_tmp = log_loss(list(total_tests['label']), submit_score_clip_tmp)

print("base final_score = " + str(final_score))
print("base final_score clip = " + str(final_score_clip))