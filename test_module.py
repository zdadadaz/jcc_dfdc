#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:45:15 2020

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

# import seaborn as sns
# import matplotlib.pylab as plt


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

class Test():
    def __init__(self, arg):
        self.save_interval = arg['save_interval']
        self.margin = arg['margin']
        self.dirname = arg['dirname']
        self.dir_json = arg['dir_json']
        self.name = arg['name']
        self.submission = None
        self.final_score = None
        self.final_score_clip = None
        
    def prepare_data(self):
        dir_json = self.dir_json
        files =[]
        with open(dir_json) as json_file:
            data = json.load(json_file)
            files = pd.DataFrame.from_dict(data, orient='index')
            files.reset_index(level=0, inplace=True)
        real = resample(files[files['label']=='REAL'], n_samples=200, replace=False, random_state=800)
        fake = resample(files[files['label']=='FAKE'], n_samples=200, replace=False, random_state=100)
        total_tests = real.append(fake)
        total_tests.reset_index(level=0, inplace=True,drop=True)
        return total_tests
    
    def predict(self, classifier, datasets, preprocess= False, save = False):
        submit = []
        save_interval = self.save_interval
        total_num = len(datasets)
        for v in range(total_num):
            print("index: "+str(v)+" /"+ str(total_num))
            vi = datasets.iloc[v][0]
            re_video = 0.5
            try:
                # read video
                cur_size = (1080,1920)
                crop_size_iist = [(720, 1280), (1080,1920)]
                for cr in range(2):
                    crop_size = crop_size_iist[cr]
                    reader = cv2.VideoCapture(os.path.join(self.dirname, vi))
                    images = []
                    for i in range(int(reader.get(cv2.CAP_PROP_FRAME_COUNT))):
                        _, image = reader.read()
                        if i % save_interval != 0:
                            continue
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        if cr == 0:
                            image = image[(int(cur_size[1]/2)-int(crop_size[1]/2)): (int(cur_size[1]/2)+int(crop_size[1]/2)),0: int(crop_size[0]), :]
                        images.append(image)
                    reader.release()
                    images = np.stack(images)
                    re_imgs = []
                    # detect face 
                    faces = self.face_detect(images)
                    if len(faces) == len(images) or cr == 1:
                        break
                if save:
                    for i in range(len(faces)):
                        imageio.imwrite(os.path.join('./tmp_mtcnn',vi+"_"+str(i)+"_"+str(0)+".jpg"),faces[i],'jpg')
                if preprocess:
                    faces = self.transform(faces)
                re_imgs = classifier.predict(np.array(faces))
                re_video = np.average(re_imgs)
                if np.isnan(re_video):
                    re_video = 0.5
                print("score = "+str(re_video))
            except:
                re_video = 0.5
                print("score = "+str(re_video))
            submit.append([vi,1.0-re_video])
        
        submit_score = [[i[1], 1-i[1]] for i in submit]
        final_score = log_loss(list(datasets['label']), submit_score)
        
        submit_score_clip = [[self.clip(i[1]), 1-self.clip(i[1])] for i in submit]
        final_score_clip = log_loss(list(datasets['label']), submit_score_clip)
        
        submission = pd.DataFrame(submit, columns=['filename', 'label']).fillna(0.5)
        submission.sort_values('filename').to_csv('submission_'+ self.name +'.csv', index=False)
        
        print(self.name + " final_score = " + str(final_score))
        print(self.name + " final_score clip = " + str(final_score_clip))
        self.submission = submission
        self.final_score = final_score
        self.final_score_clip = final_score_clip
        return submission, final_score
    
    def face_detect(self, images):
        faces = []
        return faces
    
    def clip(self, value, shift = 0.1):
        if value < shift:
            return shift
        if value >(1-shift):
            return (1-shift)
        return value
    def transform(self, image):
        pass
    
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
    
            
class Test_mtcnn(Test):
    def __init__(self, arg):
        super().__init__(arg)
        self.detector = MTCNN()
    def face_detect(self, images):
        faces = []
        for image in images:
            boxes = self.detector.detect_faces(image)
            if len(boxes) == 0:
                continue
            box = boxes[0]['box']
            face_position = [0]*4
            maxframe = max(int(box[3]/2),int(box[2]/2))
            center_y = box[1]+int(box[3]/2)
            center_x = box[0]+int(box[2]/2)
            face_position[0] = center_y-maxframe
            face_position[2] = center_y+maxframe
            face_position[3] = center_x-maxframe
            face_position[1] = center_x+maxframe
            offset = round(self.margin * (face_position[2] - face_position[0]))
            y0 = max(face_position[0] - offset, 0)
            x1 = min(face_position[1] + offset, image.shape[1])
            y1 = min(face_position[2] + offset, image.shape[0])
            x0 = max(face_position[3] - offset, 0)
            face = image[y0:y1,x0:x1]
            if sum(np.array(face.shape)==0) == 1:
                continue
            face = cv2.resize(face,(256,256))/255.
            faces.append(face)
        return faces
    
    def face_detect_fast(self, images):
        faces = []
        cur_center=None
        crop_size_iist = [(500, 500), (1080,1920)]
        crop_size = crop_size_iist[0]
        for image in images:
            if len(faces) != 0:
                y0 = max(0,(cur_center[1]-int(crop_size[1]/2)))
                y1 = min(1920, (cur_center[1]+int(crop_size[1]/2)))
                x0 = max(0, (cur_center[0]-int(crop_size[0]/2)))
                x1 = min(1080, (cur_center[0]+int(crop_size[0]/2)))
                image = image[y0:y1, x0:x1, :]
            boxes = self.detector.detect_faces(image)
            if len(boxes) == 0:
                continue
            box = boxes[0]['box']
            face_position = [0]*4
            maxframe = max(int(box[3]/2),int(box[2]/2))
            center_y = box[1]+int(box[3]/2)
            center_x = box[0]+int(box[2]/2)
            face_position[0] = center_y-maxframe
            face_position[2] = center_y+maxframe
            face_position[3] = center_x-maxframe
            face_position[1] = center_x+maxframe
            offset = round(self.margin * (face_position[2] - face_position[0]))
            y0 = max(face_position[0] - offset, 0)
            x1 = min(face_position[1] + offset, image.shape[1])
            y1 = min(face_position[2] + offset, image.shape[0])
            x0 = max(face_position[3] - offset, 0)
            face = image[y0:y1,x0:x1]
            if sum(np.array(face.shape)==0) == 1:
                continue
            if len(faces) == 1:
                cur_center= [center_x, center_y]
            else:
                diff_x = int(crop_size[0]/2) - center_x
                diff_y = int(crop_size[1]/2) - center_y
                cur_center = [cur_center[0]+diff_x,cur_center[1]+diff_y]
                
                cur_center = [cur_center[0]+center_x,center_y]
            face = cv2.resize(face,(256,256))/255.
            faces.append(face)
        return faces

class Test_base(Test):
    def __init__(self, arg):
        super().__init__(arg)
    def face_detect(self, images):
        faces = []
        for img in images:
            face_positions = face_recognition.face_locations(img)
            for face_position in face_positions:
                offset = round(self.margin * (face_position[2] - face_position[0]))
                y0 = max(face_position[0] - offset, 0)
                x1 = min(face_position[1] + offset, img.shape[1])
                y1 = min(face_position[2] + offset, img.shape[0])
                x0 = max(face_position[3] - offset, 0)
                face = img[y0:y1,x0:x1]
                inp = cv2.resize(face,(256,256))/255.
                faces.append(inp)
        return faces


class Test_base_fft(Test_base):
    def __init__(self, arg):
        super().__init__(arg)
    
    def transform(self, images):
        faces = []
        for im in images:
            faces.append(self.dft2(im))
        return faces
    

class Test_mtcnn_fft(Test_mtcnn):
    def __init__(self, arg):
        super().__init__(arg)
    
    def transform(self, images):
        faces = []
        for im in images:
            faces.append(self.dft2(im))
        return faces 

    
# =============================================================================
# coefficient
# =============================================================================
arg = {}
arg['name'] = "base_fft"
arg['save_interval'] = 150
arg['margin']=0.2
arg['dirname'] = "./../fb_whole/dfdc_train_part_21"
arg['dir_json'] = './../fb_whole/metadata_21.json'

arg_m = {}
arg_m['name'] = "mtcnn_fft"
arg_m['save_interval'] = 75
arg_m['margin']=0
arg_m['dirname'] = "./../fb_whole/dfdc_train_part_21"
arg_m['dir_json'] = './../fb_whole/metadata_21.json'

# =============================================================================
# declare module class
# =============================================================================
test_base = Test_base_fft(arg)
test_mtcnn = Test_mtcnn_fft(arg_m)

# classifier = MesoInception4()
# classifier.load('weight_tmp/mesoInc4_aug-26-0.38.hdf5')
# # classifier.load('result/meso4inc_base/v1.0.2/meso4inc-04-0.74.hdf5')


# =============================================================================
# Testing
# =============================================================================
# data = test_base.prepare_data()

# test_mtcnn.predict(classifier, data)
# test_base.predict(classifier, data)

classifier = MesoInception4()
classifier.load('weight_tmp/mesoInc4_fft-26-0.57.hdf5')
data = test_base.prepare_data()

test_mtcnn.predict(classifier, data, preprocess=True)
test_base.predict(classifier, data, preprocess=True)
