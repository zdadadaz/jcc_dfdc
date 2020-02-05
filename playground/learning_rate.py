#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:48:07 2020

@author: liulara
"""

# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np

#Reference 
#https://www.pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/

class LearningRateDecay():
    def plot(self, epochs, title="Learning Rate Schedule"):
        # compute the set of learning rates for each corresponding
        # epoch
        lrs = [self(i) for i in epochs]
        # the learning rate schedule
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(epochs, lrs)
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")
        
class StepDecay(LearningRateDecay):
    def __init__(self, initAlpha=0.001, factor=0.1, dropEvery=1):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery
    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = float(self.initAlpha * (self.factor ** exp))
        alpha = 1e-6 if alpha < float(1e-6) else alpha
		# return the learning rate
        return float(alpha)

schedule = StepDecay(initAlpha=1e-3, factor=0.1, dropEvery=1)
# callbacks = [LearningRateScheduler(schedule)]
# history.history['lr']
schedule.plot([i for i in range(10)])
        
    