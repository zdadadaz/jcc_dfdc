import numpy as np
from classifiers import *
from pipeline import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
import cv2
import random


classifier = MesoInception4()
# classifier.load('org_weights/M')

# model = classifier.init_model()
# print(model.summary())

epochs=40
batch_size=50

def blur(img):
    if random.random() <0.2:
        return (cv2.blur(img,(5,5))) 
    else:
        return img
    
def dft2(img):
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

dataGenerator = ImageDataGenerator(rescale=1./255,
                                   preprocessing_function = dft2 )
                                    # horizontal_flip=True,
                                    # # width_shift_range=50,
                                    # # height_shift_range=50,
                                    # zoom_range =0.2,
                                    # preprocessing_function = blur)
val_dataGenerator = ImageDataGenerator(rescale=1./255,
                                       preprocessing_function = dft2)
# dataGenerator = ImageDataGenerator(rescale=1./255,validation_split=0.1)
generator = dataGenerator.flow_from_directory(
        # 'deepfake_database/train_test',
        'db_small/train',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary',
        subset='training')
val_generator = val_dataGenerator.flow_from_directory(
        # 'deepfake_database/validation',
        'db_small/val',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary')
# val_generator = dataGenerator.flow_from_directory(
#         'deepfake_database/train_test',
#         # 'fb_db/train',
#         target_size=(256, 256),
#         batch_size=batch_size,
#         class_mode='binary',
#         subset='validation')

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

# checkpoint
filepath="./weight_tmp/meso4inc-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = classifier.fit_generator(generator,int(generator.samples/batch_size),epochs, \
                        callbacks_list,val_generator,int(val_generator.samples/batch_size))
    
# assert tracker_cb.trained_epochs == [0, 1, 2, 3, 4]

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model acc')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# classifier.saveMode("model.h5")