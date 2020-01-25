import numpy as np
from pipeline import *
from mtcnn import MTCNN
from classifiers import *

import json
import pandas as pd
import cv2
import os 

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from matplotlib import pyplot
import time

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)


class Extract_image():
    def __init__(self, jsons, dirs, dir_out):
        self.detector = MTCNN()
        self.margin = 0
        self.save_interval = 60
        self.json_lists = jsons
        self.dir_lists  = dirs
        self.dir_out = dir_out
        self.train_list = self.read_metadata(jsons)
        self.total_files = len(self.train_list)
        self.existed_filenames = self.read_existed_files(dir_out)
        self.extract()
    def read_metadata(self, path_json):
        with open(path_json) as json_file:
            data = json.load(json_file)
        train_list = pd.DataFrame.from_dict(data, orient='index')
        train_list.reset_index(level=0, inplace=True)
        
        return train_list
    
    def read_existed_files(self, out_dirname):
        existed_filenames = set()
        for fake_real in listdir(out_dirname):
            for f in listdir(join(out_dirname,fake_real)):
                if  (f[-4:] == '.jpg'):        
                    existed_filenames.add(f.split('.')[0])
        return existed_filenames
    
    def extract(self):
        for i in range(1,self.total_files):
            vid = self.train_list.loc[i]['index']
            print("index : "+ str(i) + " / " + str(self.total_files))
            print("vid: "+vid)
            if vid.split('.')[0] in self.existed_filenames:
                print('Video is extracted, id: ' + vid)
                continue
            if isfile(join(self.dir_lists, vid)) and ((vid[-4:] == '.mp4') or (vid[-4:] == '.avi') or (vid[-4:] == '.mov')):
                out_filename = join(self.dir_out,self.train_list.loc[i]['label'])
                self.capture_image(vid,out_filename)
    
    def capture_image(self, vid, out_filename):
        # reader = cv2.VideoCapture(os.path.join(self.dir_lists, vid))
        # read video
        cur_size = (1080,1920)
        crop_size_iist = [(1080, 1920), (1080,1920)]
        crop_size_iist = [(720, 1280), (1080,1920)]
        for cr in range(2):
            crop_size = crop_size_iist[cr]
            reader = cv2.VideoCapture(os.path.join(self.dir_lists, vid))
            images = []
            # print(int(reader.get(cv2.CAP_PROP_FRAME_COUNT)))
            if int(reader.get(cv2.CAP_PROP_FRAME_COUNT)) % self.save_interval <= 2:
                save_interval = self.save_interval-30 # perform face detection every {save_interval} frames
            else:
                save_interval = self.save_interval
            count_image = 0
            for i in range(int(reader.get(cv2.CAP_PROP_FRAME_COUNT))):
                _, image = reader.read()
                if i % save_interval != 0 and i % save_interval != 1 and i % save_interval != 2:
                    continue
                if count_image > 18:
                    break
                count_image += 1
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if cr == 0:
                    image = image[(int(cur_size[1]/2)-int(crop_size[1]/2)): (int(cur_size[1]/2)+int(crop_size[1]/2)),0: int(crop_size[0]), :]
                images.append(image)
            reader.release()
            images = np.stack(images)
            # detect face 
            faces = self.face_detect(images)
            if len(faces) == len(images) or cr == 1:
                for i in range(len(faces)):
                    output_name= vid + "_" + str(i) +  ".jpg"
                    imageio.imwrite(join(out_filename,output_name),faces[i],'jpg')
                break
        
        
    def face_detect(self, images):
        faces = []
        for im in range(1,len(images),3):
            image = images[im]
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
            face1 = images[im-1][y0:y1,x0:x1]
            face2 = image[y0:y1,x0:x1]
            face3 = images[im+1][y0:y1,x0:x1]
            # pyplot.subplot(330 + 1 + 0)
            # pyplot.imshow(face1)
            # pyplot.subplot(330 + 1 + 1)
            # pyplot.imshow(face2)
            # pyplot.subplot(330 + 1 + 2)
            # pyplot.imshow(face3)
            if sum(np.array(face1.shape)==0) == 1 and sum(np.array(face2.shape)==0) == 1 and sum(np.array(face3.shape)==0) == 1:
                continue
            # face = cv2.resize(face,(256,256))/255.
            faces.append(face1)
            faces.append(face2)
            faces.append(face3)
        return faces


folderNames = [  "dfdc_train_part_15", "dfdc_train_part_16", "dfdc_train_part_17", "dfdc_train_part_18", "dfdc_train_part_19", "dfdc_train_part_20"]
for folderName in folderNames:
    # folderName="dfdc_train_part_21"
    
    
    path_json = "./../fb_whole/"+ folderName +"/metadata.json"
    dirname = './../fb_whole/'+folderName
    out_dirname = './fb_db/'+folderName
    
    if not os.path.isfile(out_dirname):
        cmd = 'mkdir ' + out_dirname
        os.system(cmd)
    
    tmp_dir = out_dirname + "/FAKE"
    if not os.path.isfile(tmp_dir):
        cmd = 'mkdir ' + tmp_dir
        os.system(cmd)
    
    tmp_dir = out_dirname + "/REAL"
    if not os.path.isfile(tmp_dir):
        cmd = 'mkdir ' + tmp_dir
        os.system(cmd)
    
    # start = time.time()
    Extract_image(path_json, dirname, out_dirname)
    # elapsed = time.time() - start
    # print(f', {elapsed:.3f} seconds')
