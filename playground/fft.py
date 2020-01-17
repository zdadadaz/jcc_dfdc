#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 21:31:40 2020

@author: zdadadaz
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

def dft2(img):
    out = np.zeros(img.shape)
    def dft2_onechennel(image):
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        a = np.log(np.abs(fshift))
        a = a - a.mean()
        a = (a+1e-9) / (a.max()+1e-9)
        return a
    for i in range(3):
        out[:,:,i] = dft2_onechennel(img[:,:,i])
    return out

# img = cv2.imread('./../db_small/val/df/abaodketae.mp4_3.jpg')
# out = dft2(img)

img = cv2.imread('./../db_small/val/real/abpibxailk.mp4_3.jpg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
a = np.log(np.abs(fshift))
a = a - a.mean()
a = (a+1e-9) / (a.max()+1e-9)
magnitude_spectrum = a


plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


