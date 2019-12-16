import numpy as np
from classifiers import *
from pipeline import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback

classifier = MesoInception4()
# classifier.load('weights/MesoInception_DF')

epochs=10
batch_size=50

dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
        'deepfake_database/train_test',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary',
        subset='training')

# Train
class TrackerCallback(Callback):
    def __init__(self):
        # test starting from non-zero initial epoch
        self.trained_epochs = []
        self.trained_batches = []
        self.steps_per_epoch_log = []
        super(TrackerCallback, self).__init__()

    def set_params(self, params):
        super(TrackerCallback, self).set_params(params)
        self.steps_per_epoch_log.append(params['steps'])

    # define tracer callback
    def on_epoch_begin(self, epoch, logs):
        self.trained_epochs.append(epoch)

    def on_batch_begin(self, batch, logs):
        self.trained_batches.append(batch)

tracker_cb = TrackerCallback()


classifier.fit_generator(generator,int(generator.samples/batch_size),epochs,[tracker_cb])
# assert tracker_cb.trained_epochs == [0, 1, 2, 3, 4]

classifier.saveMode("model.h5")