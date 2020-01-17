#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:52:38 2020

@author: zdadadaz
"""

from classifiers import *
import cv2

def meso_preprocessing_input():
    

def extract_features(directory):
	classifier = MesoInception4(include_top = False)
    model = classifier.init_model()
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	features = dict()
 
	for name in listdir(directory):
		filename = directory + '/' + name
		image = load_img(filename, target_size=(256, 256))
		image = img_to_array(image)
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		image = preprocess_input(image)
		feature = model.predict(image, verbose=0)

		image_id = name.split('.')[0]
		features[image_id] = feature
	return features

def model(imageSize, max_length):

	input1 = Input(shape=(4096,)) # feature
	drop_1 = Dropout(0.5)(inputs1)
	dense_1 = Dense(256, activation='relu')(fe1)

	inputs2 = Input(shape=(max_length,))
	embed = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	drop_2 = Dropout(0.5)(se1)
	dense_2 = LSTM(256)(se2)

	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)

	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	model.summary()
	return model