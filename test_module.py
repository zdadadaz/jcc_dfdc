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
from keras.applications.xception import preprocess_input
import time
# import seaborn as sns
# import matplotlib.pylab as plt

from PIL import Image as pil_image

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
        self.size_for_testing = arg['size_for_testing']
        self.rescale = arg['rescale']
        # self.read_video = arg['read_video']
        
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
        # submission.sort_values('filename').to_csv('submission_'+ self.name +'.csv', index=False)
        
        print(self.name + " final_score = " + str(final_score))
        print(self.name + " final_score clip = " + str(final_score_clip))
        self.submission = submission
        self.final_score = final_score
        self.final_score_clip = final_score_clip
        return submission, final_score
    
    # def read_video(self, path):
    #     if self.read_video == "cv":
    #         video = cv2.VideoCapture(os.path.join(self.dirname, vi))
    #         number_frame = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    #     else:
    #         video = Video(os.path.join("/kaggle/input/deepfake-detection-challenge/test_videos/", vi))
    #         number_frame = video.__len__(),save_interval
    #     return video, number_frame
    
    def face_detect(self, images):
        faces = []
        return faces
    
    def clip(self, value, shift = 0.1):
        if value < (0.5-shift):
            return 0
        if value >(0.5+shift):
            return 1
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
            if maxframe<=30:
    #            too small face is wrong detection
                continue
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
            
            face = cv2.resize(face,(self.size_for_testing,self.size_for_testing),interpolation=cv2.INTER_NEAREST)/self.rescale
            # face = cv2.resize(face,(256,256))/255.
            faces.append(face)
        return faces
    
    def face_detect_fast_resize(self,images):
        faces = []
        cur_center=[750, 960]
        crop_size_iist = [(500, 500), (1080,1080)]
        resize = (256,256)
        crop_size = crop_size_iist[1]
        for image in images:
    #        Crop to 1080x1080 and resize to 256x256
            image_m = image[ :,(960-int(crop_size[1]/2)):(960+int(crop_size[1]/2)), :]
            image_m = cv2.resize(image_m,(resize[0],resize[1]))
            boxes = self.detector.detect_faces(image_m)
            if len(boxes) == 0:
    #            should jump out and go for normal size
                break
            box = boxes[0]['box']
            
            rescale_box = [0]*4
            rescale_box[0] = int(box[0]/resize[0]* crop_size[0]) + (960-int(crop_size[1]/2))
            rescale_box[1] = int(box[1]/resize[1]* crop_size[1]) 
            rescale_box[2] = int(box[2]/resize[0]* crop_size[0]) 
            rescale_box[3] = int(box[3]/resize[1]* crop_size[1])
            box = rescale_box
            
            face_position = [0]*4
            maxframe = max(int(box[3]/2),int(box[2]/2))
            if maxframe<=30:
    #            too small face is wrong detection
                continue
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
                break
            
            face = cv2.resize(face,(self.size_for_testing,self.size_for_testing),interpolation=cv2.INTER_NEAREST)/self.rescale
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
                inp = cv2.resize(face,(self.size_for_testing,self.size_for_testing),interpolation=cv2.INTER_NEAREST)/self.rescale
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

class Test_mtcnn_xception(Test_mtcnn):
    def __init__(self, arg):
        super().__init__(arg)
    def transform(self, images):
        images = np.stack(images)
        faces = preprocess_input(images)
        return faces 
    
    def predict(self, classifier, datasets, preprocess= False, save = False):
        submit = []
        save_interval = self.save_interval
        total_num = len(datasets)
        for v in range(total_num):
            print("index: "+str(v)+" /"+ str(total_num))
            vi = datasets.iloc[v][0]
            re_video = 0.5
            # try:
            # read video
            count = 0
            reader = cv2.VideoCapture(os.path.join(self.dirname, vi))
            number_frame = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
            images = []
            for i in range(number_frame):
                _, image = reader.read()
                # image = video.get(i)
                if i % save_interval != 0:
                    continue
                if count == 10:
                    break
                count+=1
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
            reader.release()
            images = np.stack(images)
            re_imgs = []
            # detect face 
            faces = self.face_detect_fast_resize(images)
            if len(faces) != len(images):
                faces = self.face_detect(images)
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
            # except:
            #     re_video = 0.5
            #     print("some error")
            #     print("score = "+str(re_video))
            # submit.append([vi,re_video])
            submit.append([vi,1.0-re_video])
        
        submit_score = [[i[1], 1-i[1]] for i in submit]
        final_score = log_loss(list(datasets['label']), submit_score)
        
        submit_score_clip = [[self.clip(i[1]), 1-self.clip(i[1])] for i in submit]
        final_score_clip = log_loss(list(datasets['label']), submit_score_clip)
        
        submission = pd.DataFrame(submit, columns=['filename', 'label']).fillna(0.5)
        # submission.sort_values('filename').to_csv('submission_'+ self.name +'.csv', index=False)
        
        print(self.name + " final_score = " + str(final_score))
        print(self.name + " final_score clip = " + str(final_score_clip))
        self.submission = submission
        self.final_score = final_score
        self.final_score_clip = final_score_clip
        return submission, final_score

class Test_mtcnn_xception_tcn(Test_mtcnn):
    def __init__(self, arg):
        super().__init__(arg)
    def transform(self, images):
        images = np.stack(images)
        faces = preprocess_input(images)
        return faces 
    
    def predict(self, classifier, classifier_notop, classifier_tcn, datasets, preprocess= False, save = False):
        submit = []
        save_interval = self.save_interval
        total_num = len(datasets)
        face_number = 10
        for v in range(total_num):
            print("index: "+str(v)+" /"+ str(total_num))
            vi = datasets.iloc[v][0]
            re_video = 0.5
            # try:
            # read video
            count = 0
            reader = cv2.VideoCapture(os.path.join(self.dirname, vi))
            number_frame = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
            images = []
            for i in range(number_frame):
                _, image = reader.read()
                # image = video.get(i)
                # if i % save_interval != 0 and i % save_interval != 1 and i % save_interval != 2:
                    # continue
                if i % save_interval != 0:
                    continue
                if count == 10:
                    break
                count+=1
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
            reader.release()
            images = np.stack(images)
            re_imgs = []
            # detect face 
            faces = self.face_detect_fast_resize(images)
            if len(faces) != len(images):
                faces = self.face_detect(images)
            if len(faces)>face_number:
                faces = faces[:face_number]
            if save:
                for i in range(len(faces)):
                    imageio.imwrite(os.path.join('./tmp_mtcnn',vi+"_"+str(i)+"_"+str(0)+".jpg"),faces[i],'jpg')
            if preprocess:
                faces = self.transform(faces)
            if len(faces)<face_number:
                out = classifier.predict(np.array(faces))
                # print("xception")
            else:
                re_imgs = classifier_notop.predict(np.array(faces))
                re_imgs = np.expand_dims(re_imgs, axis=0)
                out = classifier_tcn.predict(re_imgs)
            # print("xception+lstm")
            re_video = np.average(out)
            if np.isnan(re_video):
                re_video = 0.5
            print("score = "+str(re_video))
            submit.append([vi,1.0-re_video])
        
        submit_score = [[i[1], 1-i[1]] for i in submit]
        final_score = log_loss(list(datasets['label']), submit_score)
        
        submit_score_clip = [[self.clip(i[1]), 1-self.clip(i[1])] for i in submit]
        final_score_clip = log_loss(list(datasets['label']), submit_score_clip)
        
        submission = pd.DataFrame(submit, columns=['filename', 'label']).fillna(0.5)
        # submission.sort_values('filename').to_csv('submission_'+ self.name +'.csv', index=False)
        
        print(self.name + " final_score = " + str(final_score))
        print(self.name + " final_score clip = " + str(final_score_clip))
        self.submission = submission
        self.final_score = final_score
        self.final_score_clip = final_score_clip
        return submission, final_score

    # def face_detect(self, images):
    #     faces = []
    #     for im in range(0,len(images),3):
    #         image = images[im]
    #         boxes = self.detector.detect_faces(image)
    #         if len(boxes) == 0:
    #             continue
    #         box = boxes[0]['box']
    #         face_position = [0]*4
    #         maxframe = max(int(box[3]/2),int(box[2]/2))
    #         if maxframe<=30:
    # #            too small face is wrong detection
    #             continue
    #         center_y = box[1]+int(box[3]/2)
    #         center_x = box[0]+int(box[2]/2)
    #         face_position[0] = center_y-maxframe
    #         face_position[2] = center_y+maxframe
    #         face_position[3] = center_x-maxframe
    #         face_position[1] = center_x+maxframe
    #         offset = round(self.margin * (face_position[2] - face_position[0]))
    #         y0 = max(face_position[0] - offset, 0)
    #         x1 = min(face_position[1] + offset, image.shape[1])
    #         y1 = min(face_position[2] + offset, image.shape[0])
    #         x0 = max(face_position[3] - offset, 0)
    #         face = image[y0:y1,x0:x1]
    #         face1 = images[im+1][y0:y1,x0:x1]
    #         face2 = images[im+2][y0:y1,x0:x1]
    #         if sum(np.array(face.shape)==0) == 1 and sum(np.array(face1.shape)==0) == 1 and sum(np.array(face2.shape)==0) == 1:
    #             continue
    #         face = cv2.resize(face,(self.size_for_testing,self.size_for_testing),interpolation=cv2.INTER_NEAREST)/self.rescale
    #         face1 = cv2.resize(face1,(self.size_for_testing,self.size_for_testing),interpolation=cv2.INTER_NEAREST)/self.rescale
    #         face2 = cv2.resize(face2,(self.size_for_testing,self.size_for_testing),interpolation=cv2.INTER_NEAREST)/self.rescale
            
    #         faces.append(face)
    #         faces.append(face1)
    #         faces.append(face2)
    #     return faces
    
    # def face_detect_fast_resize(self,images):
    #     faces = []
    #     cur_center=[750, 960]
    #     crop_size_iist = [(500, 500), (1080,1080)]
    #     resize = (256,256)
    #     crop_size = crop_size_iist[1]
    #     for im in range(0,len(images),3):
    #         image = images[im]
    # #        Crop to 1080x1080 and resize to 256x256
    #         image_m = image[ :,(960-int(crop_size[1]/2)):(960+int(crop_size[1]/2)), :]
    #         image_m = cv2.resize(image_m,(resize[0],resize[1]))
    #         boxes = self.detector.detect_faces(image_m)
    #         if len(boxes) == 0:
    # #            should jump out and go for normal size
    #             break
    #         box = boxes[0]['box']
            
    #         rescale_box = [0]*4
    #         rescale_box[0] = int(box[0]/resize[0]* crop_size[0]) + (960-int(crop_size[1]/2))
    #         rescale_box[1] = int(box[1]/resize[1]* crop_size[1]) 
    #         rescale_box[2] = int(box[2]/resize[0]* crop_size[0]) 
    #         rescale_box[3] = int(box[3]/resize[1]* crop_size[1])
    #         box = rescale_box
            
    #         face_position = [0]*4
    #         maxframe = max(int(box[3]/2),int(box[2]/2))
    #         if maxframe<=30:
    # #            too small face is wrong detection
    #             continue
    #         center_y = box[1]+int(box[3]/2)
    #         center_x = box[0]+int(box[2]/2)
    #         face_position[0] = center_y-maxframe 
    #         face_position[2] = center_y+maxframe
    #         face_position[3] = center_x-maxframe
    #         face_position[1] = center_x+maxframe
                     
    #         offset = round(self.margin * (face_position[2] - face_position[0]))
    #         y0 = max(face_position[0] - offset, 0)
    #         x1 = min(face_position[1] + offset, image.shape[1])
    #         y1 = min(face_position[2] + offset, image.shape[0])
    #         x0 = max(face_position[3] - offset, 0)
    #         face = image[y0:y1,x0:x1]
    #         face1 = images[im+1][y0:y1,x0:x1]
    #         face2 = images[im+2][y0:y1,x0:x1]
    #         if sum(np.array(face.shape)==0) == 1:
    #             break
            
    #         face = cv2.resize(face,(self.size_for_testing,self.size_for_testing),interpolation=cv2.INTER_NEAREST)/self.rescale
    #         face1 = cv2.resize(face1,(self.size_for_testing,self.size_for_testing),interpolation=cv2.INTER_NEAREST)/self.rescale
    #         face2 = cv2.resize(face2,(self.size_for_testing,self.size_for_testing),interpolation=cv2.INTER_NEAREST)/self.rescale
            
    #         faces.append(face)
    #         faces.append(face1)
    #         faces.append(face2)
    #     return faces


# =============================================================================
# coefficient
# =============================================================================
arg = {}
arg['name'] = "meso"
arg['save_interval'] = 30
arg['margin']=0.2
arg['dirname'] = "./../fb_whole/dfdc_train_part_24"
arg['dir_json'] = './../fb_whole/metadata_24.json'
arg['size_for_testing'] = 256
arg['rescale'] = 255.0

# arg = {}
# arg['name'] = "mtcnn"
# arg['save_interval'] = 75
# arg['margin']=0
# arg['dirname'] = "./../fb_whole/dfdc_train_part_24"
# arg['dir_json'] = './../fb_whole/metadata_24.json'
# arg['size_for_testing'] = 299
# arg['rescale'] = 1.0

arg_m = {}
arg_m['name'] = "mtcnn_fast"
arg_m['save_interval'] = 30
arg_m['margin']=0
arg_m['dirname'] = "./../fb_whole/dfdc_train_part_24"
arg_m['dir_json'] = './../fb_whole/metadata_24.json'
arg_m['size_for_testing'] = 299
arg_m['rescale'] = 1.0
# =============================================================================
# declare module class
# =============================================================================
# test_base = Test_base_fft(arg)
test_mtcnn = Test_mtcnn(arg)
# test_mtcnn_fast = Test_mtcnn_xception(arg_m)
test_mtcnn_fast_xtcn = Test_mtcnn_xception_tcn(arg_m)
# classifier = MesoInception4()
# classifier.load('weight_tmp/mesoInc4_aug-26-0.38.hdf5')
# classifier.load('result/meso4inc_base/v1.0.3/mesoInc4_fft-25-0.37.hdf5')


# =============================================================================
# Testing
# =============================================================================
classifier = Xception_main()
classifier.load('./result/xception/x1.1.1/xception-02-0.34.hdf5')
data = test_mtcnn_fast_xtcn.prepare_data()

times = []
# start = time.time()
# test_mtcnn.predict(classifier, data, preprocess=False)
# elapsed = time.time() - start
# times.append(elapsed)
# print("elapsed: ", str(elapsed))

# start = time.time()
# test_mtcnn.predict(classifier, data, preprocess=False, save = False)
# elapsed = time.time() - start
# print("elapsed: ", str(elapsed))
# # test_base.predict(classifier, data, preprocess=True)
# times.append(elapsed)

classifier_notop = Xception_main_noTop()
classifier_tcn = bitslm_main()
# classifier_tcn.load('./result/xception/xtc1.0.0/xception_tcn-24-0.16.hdf5')
# classifier_tcn.load('./result/xception/xtc1.0.2/xception_tcn-04-0.31.hdf5')
classifier_tcn.load('./result/xception/xlstm1.0.1/bitslm_main-07-0.28.hdf5')
start = time.time()
test_mtcnn_fast_xtcn.predict(classifier, classifier_notop, classifier_tcn, data, preprocess=True, save = False)
elapsed = time.time() - start
print("elapsed: ", str(elapsed))
# test_base.predict(classifier, data, preprocess=True)
times.append(elapsed)


# aa = test_mtcnn_fast.submission
# aa['correct'] = ((aa['label']>0.5) ^ (data['label']=="FAKE"))
# aa['correct'] =  ~aa['correct']

