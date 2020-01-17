#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 21:15:46 2020

@author: zdadadaz
"""

import numpy as np
import cv2
import random
from matplotlib import pyplot as plt

def blur(img):
    if random.random() <0.1:
        sig = random.random()*3
        return (cv2.GaussianBlur(img,(5,5),sig)) 
    else:
        return img
    
def compress(img):
    q = random.random()*70 + 30
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg
    
image = cv2.imread('./db_small/val/real/abpibxailk.mp4_3.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.imshow(image)
# plt.imshow()
image1 = compress(image)
# plt.imshow(image1)

plt.subplot(121),plt.imshow(image)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(image1)
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


