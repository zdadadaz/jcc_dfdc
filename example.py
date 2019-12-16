import numpy as np
from classifiers import *
from pipeline import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.preprocessing.image import ImageDataGenerator

# 1 - Load the model and its pretrained weights
# classifier = Meso4()
# classifier.load('weights/model.h5')

classifier = MesoInception4()
classifier.load('weights/model.h5')

# 2 - Minimial image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
        # 'test_images',
        'deepfake_database/validation',
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary',
        subset='training')

# 3 - Predict
forged = 0
real = 0
forged_tot = 0
real_tot = 0

for i in range(generator.samples):
    X, y = generator.next()
    predicted = classifier.predict(X)
    if int(y) == 1:
        real_tot += 1
    else:
        forged_tot += 1

    if int(predicted > 0.5) == 1 and int(y) == 1:
        real += 1
    elif int(predicted > 0.5) == 0 and int(y) == 0:
        forged += 1
    # print('Predicted :', predicted, '\nReal class :', y)
    # imgplot = plt.imshow(X[0])
    # plt.show()


print('Forged classification score:', str(forged/forged_tot))
print('Real classification score:', str(real/real_tot))
print('Total classification score:', str((real+forged)/(forged_tot+real_tot)))

# 4 - Prediction for a video dataset

# classifier.load('weights/Meso4_F2F')

# predictions = compute_accuracy(classifier, 'test_videos')
# for video_name in predictions:
#     print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])