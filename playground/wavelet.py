#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 21:50:48 2020

@author: zdadadaz
"""

import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data


# Load image
# original = pywt.data.camera()

# # Wavelet transform of image, and plot approximation and details
# titles = ['Approximation', ' Horizontal detail',
#           'Vertical detail', 'Diagonal detail']
# coeffs2 = pywt.dwt2(original, 'bior1.3')
# LL, (LH, HL, HH) = coeffs2
# fig = plt.figure(figsize=(12, 3))
# for i, a in enumerate([LL, LH, HL, HH]):
#     ax = fig.add_subplot(1, 4, i + 1)
#     ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
#     ax.set_title(titles[i], fontsize=10)
#     ax.set_xticks([])
#     ax.set_yticks([])

# fig.tight_layout()
# plt.show()

from matplotlib import pyplot
import cv2
save_interval = 60
folderName="dfdc_train_part_21"

cur_size = (1080,1920)
crop_size = (720, 1280)
# crop_size = (720, 960)
path_json = "./../fb_whole/"+ folderName +"/metadata.json"
dirname = './../../fb_whole/'+folderName+ "/tpotlnaogw.mp4"

reader = cv2.VideoCapture(dirname)
images = []
for i in range(int(reader.get(cv2.CAP_PROP_FRAME_COUNT))):
    _, image = reader.read()
    if i % save_interval != 0 and i % save_interval != 1 and i % save_interval != 2:
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[0: int(crop_size[0]), int(cur_size[1]/2)-int(crop_size[1]/2): int(cur_size[1]/2)+int(crop_size[1]/2)]
    # pyplot.subplot(330 + 1 + i)
    plt.figure()
    pyplot.imshow(image)
    images.append(image)
reader.release()

