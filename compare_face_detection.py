#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:04:31 2020

@author: zdadadaz
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
import time
from mtcnn import MTCNN
import face_recognition

from classifiers import *
#classifier = MesoInception4()
#classifier.load('model_bs_ep20.h5')
#%matplotlib inline
detector = MTCNN()


def plot_faces(images, figsize=(10.8/2, 19.2/2)):
    shape = images[0].shape
    images = images[np.linspace(0, len(images)-1, 16).astype(int)]
    im_plot = []
    for i in range(0, 16, 4):
        im_plot.append(np.concatenate(images[i:i+4], axis=0))
    im_plot = np.concatenate(im_plot, axis=1)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(im_plot)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    ax.grid(False)
    fig.tight_layout()

def timer(detector, detect_fn, images, *args):
    start = time.time()
    faces = detect_fn(detector, images, *args)
    elapsed = time.time() - start
#    print(f', {elapsed:.3f} seconds')
    print("elapsed: ", str(elapsed))
    return faces, elapsed

def detect_mtcnn(detector, images):
    faces = []
    count = 1
    for image in images:
        boxes = detector.detect_faces(image)
        if len(boxes) == 0:
            continue
        box = boxes[0]['box']
        face = image[box[1]:box[3]+box[1], box[0]:box[2]+box[0]]
        if sum(np.array(face.shape)==0) == 1:
            continue
        # count += 1
        # face = cv2.resize(face,(256,256))/255.
        # score = classifier.predict(np.array([inp]))
        # print("score: " + str(score))
        faces.append(face)
    return faces

def face_detect(detector, images,margin= 0):
    faces = []
    for image in images:
        boxes = detector.detect_faces(image)
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
        offset = round(margin * (face_position[2] - face_position[0]))
        y0 = max(face_position[0] - offset, 0)
        x1 = min(face_position[1] + offset, image.shape[1])
        y1 = min(face_position[2] + offset, image.shape[0])
        x0 = max(face_position[3] - offset, 0)
        face = image[y0:y1,x0:x1]
        if sum(np.array(face.shape)==0) == 1:
            continue
        face = cv2.resize(face,(299,299))/255.
        # face = cv2.resize(face,(256,256))/255.
        faces.append(face)
    return faces

def face_detect_fast(detector, images, margin=0):
    faces = []
    cur_center=[750, 960]
    crop_size_iist = [(500, 500), (1080,1080)]
    resize = (256,256)
    crop_size = crop_size_iist[1]
    for image in images:
#        Crop to 1080x1080 and resize to 256x256
        image_m = image[ :,(960-int(crop_size[1]/2)):(960+int(crop_size[1]/2)), :]
        image_m = cv2.resize(image_m,(resize[0],resize[1]))
        boxes = detector.detect_faces(image_m)
        if len(boxes) == 0:
#            should jump out and go for normal size
            continue
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
        
        offset = round(margin * (face_position[2] - face_position[0]))
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



times_mtcnn = []

def detect_face_recognition(images):
    faces = []
    margin = 0.2
    start = time.time()
    for image in images:
        face_positions = face_recognition.face_locations(image)
        for face_position in face_positions:
            offset = round(margin * (face_position[2] - face_position[0]))
            y0 = max(face_position[0] - offset, 0)
            x1 = min(face_position[1] + offset, image.shape[1])
            y1 = min(face_position[2] + offset, image.shape[0])
            x0 = max(face_position[3] - offset, 0)
            face = image[y0:y1,x0:x1]
            inp = cv2.resize(face,(160,160))/255.
            faces.append(inp)
    elapsed = time.time() - start
    # print(f', {elapsed:.3f} seconds')
    return faces, elapsed

sample = "./../fb_whole/dfdc_train_part_24/aadubxfxhr.mp4"

reader = cv2.VideoCapture(sample)
images_1080_1920 = []
images_720_1280 = []
images_540_960 = []
count = 0
for i in tqdm(range(int(reader.get(cv2.CAP_PROP_FRAME_COUNT)))):
    _, image = reader.read()
    count+= 1
    if count % 60 != 0:
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(type(image))
    images_1080_1920.append(image)
    images_720_1280.append(cv2.resize(image, (1280, 720)))
    images_540_960.append(cv2.resize(image, (960, 540)))
reader.release()

images_1080_1920 = np.stack(images_1080_1920)
images_720_1280 = np.stack(images_720_1280)
images_540_960 = np.stack(images_540_960)

from keras.applications.xception import preprocess_input
out = preprocess_input(images_540_960)

# MTCNN
# print('Detecting faces in 540x960 frames', end='')
# faces, elapsed = timer(detector, face_detect, images_540_960)
# times_mtcnn.append(elapsed)
#
#print('Detecting faces in 720x1280 frames', end='')
#faces, elapsed = timer(detector, face_detect, images_720_1280)
#times_mtcnn.append(elapsed)
#
#print('Detecting faces in 1080x1920 frames', end='')
#faces, elapsed = timer(detector, face_detect, images_1080_1920)
#times_mtcnn.append(elapsed)

# plot_faces(np.stack([cv2.resize(face, (160, 160)) for face in faces]))


# MTCNN crop
#print('Detecting faces in 540x960 frames', end='')
#_, elapsed = timer(detector, face_detect_fast, images_540_960)
#times_mtcnn.append(elapsed)
#
#print('Detecting faces in 720x1280 frames', end='')
#_, elapsed = timer(detector, face_detect_fast, images_720_1280)
#times_mtcnn.append(elapsed)

# print('Detecting faces in 1080x1920 frames', end='')
# faces, elapsed = timer(detector, face_detect_fast, images_1080_1920)
# times_mtcnn.append(elapsed)

# plot_faces(np.stack([cv2.resize(face, (160, 160)) for face in faces]))


# face_recognition
# print('Detecting faces in 540x960 frames', end='')
# _, elapsed = detect_face_recognition( images_540_960)
# times_mtcnn.append(elapsed)

# print('Detecting faces in 720x1280 frames', end='')
# _, elapsed = detect_face_recognition(images_720_1280)
# times_mtcnn.append(elapsed)

# print('Detecting faces in 1080x1920 frames', end='')
# faces, elapsed = detect_face_recognition( images_1080_1920)
# times_mtcnn.append(elapsed)

# plot_faces(np.stack([cv2.resize(face, (160, 160)) for face in faces]))