#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 21:47:46 2020

@author: zdadadaz
"""

# example of brighting image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load the image
# img = load_img('./../db_playground/df/114_42.jpg')
img = load_img("./../../dataset/db_small/val/real/abpibxailk.mp4_3.jpg")
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# # create image data augmentation generator
# datagen = ImageDataGenerator(brightness_range=[0.4,1.8])
# # prepare iterator
# it = datagen.flow(samples, batch_size=1)
# # generate samples and plot
# for i in range(9):
# 	# define subplot
# 	pyplot.subplot(330 + 1 + i)
# 	# generate batch of images
# 	batch = it.next()
# 	# convert to unsigned integers for viewing
# 	image = batch[0].astype('uint8')
# 	# plot raw pixel data
# 	pyplot.imshow(image)
# # show the figure
# pyplot.show()

import random
import numpy as np
import cv2
# create image data augmentation generator
def dft2( img):
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

def blur_compress( img):
    # if random.random()<0.1:
    image = blur(img.astype("float"))
    return compress(image)
    # else:
    #     return img

def blur(img):
    sig = random.random()*3
    return cv2.GaussianBlur(img,(5,5),sig)

def compress( img):
    q = random.random()*70 + 30
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg.astype("float")

def _dft2_onechennel(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    a = np.log(np.abs(fshift)+1e-9)
    a = a - np.mean(a)
    std = np.std(a)
    if std is not None:
        a /= std
    a = (a+1e-9) / (np.max(np.abs(a))+1e-9)
    return a

def _preprocess_fft( image):
    out = []
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ft = _dft2_onechennel(gray)
    image = preprocess_input(image)
    for i in range(3):
        out.append(image[:,:,1])
    out.append(ft)
    out = np.stack(out)
    out = np.transpose(out, (1, 2, 0))
    return out
    


from keras.applications.xception import preprocess_input
 
datagen = ImageDataGenerator(preprocessing_function = _preprocess_fft)
#datagen = ImageDataGenerator(rescale=1./255,preprocessing_function = blur_compress)

# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
for i in range(9):
 	# define subplot
 	pyplot.subplot(330 + 1 + i)
 	# generate batch of images
 	batch = it.next()
 	# convert to unsigned integers for viewing
 	image = batch[0]
# 	image = ((batch[0].astype('float') )*256 - 1).astype('uint8')
# 	image = batch[0].astype('uint8')
 	# plot raw pixel data
 	pyplot.imshow(image)
# show the figure
pyplot.show()

# model = Sequential()
# model.add(Dense(1))
# fit(it)