# -*- coding:utf-8 -*-
import keras

from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Concatenate, LeakyReLU, GlobalMaxPooling2D, Reshape

from keras.optimizers import Adam, SGD
from models.SpatialPyramidPooling import SpatialPyramidPooling

from keras.applications import MobileNet, Xception
from keras.applications.mobilenet import preprocess_input

IMGWIDTH = 256

class Classifier:
    def __init__():
        self.model = 0
    
    def predict(self, x):
        return self.model.predict(x)
    
    def fit(self, x, y):
        return self.model.train_on_batch(x, y)
    
    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)
    
    def load(self, path):
        self.model.load_weights(path)

    def fit_generator(self,generator,steps_per_epoch, epochs,callbacks, val_gen, val_step_per_epochs, use_multiprocessing, workers):
        return self.model.fit_generator(generator,steps_per_epoch=steps_per_epoch, epochs=epochs,callbacks=callbacks, validation_data=val_gen, validation_steps=val_step_per_epochs, use_multiprocessing=use_multiprocessing, workers=workers)
    
    def saveMode(self, outputName):
        self.model.save_weights(outputName)

class Meso1(Classifier):
    """
    Feature extraction + Classification
    """
    def __init__(self, learning_rate = 0.001, dl_rate = 1):
        self.model = self.init_model(dl_rate)
        optimizer = Adam(lr = learning_rate)
        self.model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
    
    def init_model(self, dl_rate):
        x = Input(shape = (IMGWIDTH, IMGWIDTH, 3))
        
        x1 = Conv2D(16, (3, 3), dilation_rate = dl_rate, strides = 1, padding='same', activation = 'relu')(x)
        x1 = Conv2D(4, (1, 1), padding='same', activation = 'relu')(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(8, 8), padding='same')(x1)

        y = Flatten()(x1)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)
        return KerasModel(inputs = x, outputs = y)


class Meso4(Classifier):
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
        optimizer = Adam(lr = learning_rate)
        self.model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
    
    def init_model(self): 
        x = Input(shape = (IMGWIDTH, IMGWIDTH, 3))
        
        x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        
        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return KerasModel(inputs = x, outputs = y)


class MesoInception4(Classifier):
    #MesoNet uses a batch-size of 76. The learning-rate is initially set to 10^-3 and is consecutively reduced by a factor of ten for each epoch to 10^−6
    def __init__(self, learning_rate = 0.001, include_top = True):
        self.model = self.init_model(include_top)
        optimizer = Adam(lr = learning_rate)
        # self.model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
        # optimizer = SGD(lr=learning_rate, momentum=0.9)
        self.model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
       
    def InceptionLayer(self, a, b, c, d):
        def func(x):
            x1 = Conv2D(a, (1, 1), padding='same', activation='relu')(x)
            
            x2 = Conv2D(b, (1, 1), padding='same', activation='relu')(x)
            x2 = Conv2D(b, (3, 3), padding='same', activation='relu')(x2)
            
            x3 = Conv2D(c, (1, 1), padding='same', activation='relu')(x)
            x3 = Conv2D(c, (3, 3), dilation_rate = 2, strides = 1, padding='same', activation='relu')(x3)
            
            x4 = Conv2D(d, (1, 1), padding='same', activation='relu')(x)
            x4 = Conv2D(d, (3, 3), dilation_rate = 3, strides = 1, padding='same', activation='relu')(x4)

            y = Concatenate(axis = -1)([x1, x2, x3, x4])
            
            return y
        return func
    
    def init_model(self, include_top = True):
        x = Input(shape = (IMGWIDTH, IMGWIDTH, 3))
        
        x1 = self.InceptionLayer(1, 4, 4, 2)(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = self.InceptionLayer(2, 4, 4, 2)(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)        
        
        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        
        if include_top:
            y = Flatten()(x4)
            # y = SpatialPyramidPooling([1, 2, 4])(x4)
            y = Dropout(0.5)(y)
            y = Dense(16)(y)
            y = LeakyReLU(alpha=0.1)(y)
            y = Dropout(0.5)(y)
            y = Dense(1, activation = 'sigmoid')(y)
        else:
            y = GlobalMaxPooling2D()(x4)
        
        
        return KerasModel(inputs = x, outputs = y)

class Xception_main(Classifier):
#    Reference
#    https://www.groundai.com/project/faceforensics-learning-to-detect-manipulated-facial-images/1
#    https://medium.com/@gkadusumilli/image-recognition-using-pre-trained-xception-model-in-5-steps-96ac858f4206
#     learning-rate of 0.0002 and a batch-size of 32
    def __init__(self, learning_rate = 0.0002):
        self.model = self.init_model()
        optimizer = Adam(lr = learning_rate)
        self.model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        #        set trainable layer
        for layer in self.model.layers[:20]:
            layer.trainable=False
        for layer in self.model.layers[20:]:
            layer.trainable=True
#        print(self.model.summary())
        
    def init_model(self, include_top = True):
        img_width, img_height = 224, 224
        base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top='max')
        # Top Model Block
        x = base_model.output
        y = layers.Dense(1, activation='sigmoid',use_bias=True, name='Logits')(x)
        return KerasModel(inputs=base_model.input,outputs=y)

class meso_lstm(Classifier):
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
        optimizer = Adam(lr = learning_rate)
        self.model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    def init_model(self):
        x = Input(shape = (IMGWIDTH, IMGWIDTH, 3))
        x2 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x)
        # x2 = BatchNormalization()(x2)
        # x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        # y = Flatten()(x2)
        # y = Reshape((6, 2))(x2)
        
        
class meso_wavelet(Classifier):
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
        optimizer = Adam(lr = learning_rate)
        self.model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    def init_model(self):
        x = Input(shape = (128, 128, 12))
        x2 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x)
        # x2 = BatchNormalization()(x2)
        # x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        # y = Flatten()(x2)

class MobileNet_mian(Classifier):
    def __init__(self, learning_rate = 0.001):    
        self.model = self.init_model()
        optimizer = Adam(lr = learning_rate)
        self.model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        
#        set trainable layer
        for layer in self.model.layers[:20]:
            layer.trainable=False
        for layer in self.model.layers[20:]:
            layer.trainable=True
        
    def init_model(self, include_top = True):
#       input_shape=(224,224,3), input_tensor=None,
#        base_model = MobileNet( alpha=1.0, include_top=False, weights='imagenet',  pooling='max', classes=1)
        base_model = MobileNet(weights='imagenet',require_flatten=False)
#        mobilenet last layer
        y = layers.Dense(1, activation='sigmoid',use_bias=True, name='Logits')(base_model)
        
#        meso4 last few layers
#        y = Flatten()(base_model)
#        y = Dropout(0.5)(y)
#        y = Dense(16)(y)
#        y = LeakyReLU(alpha=0.1)(y)
#        y = Dropout(0.5)(y)
#        y = Dense(1, activation = 'sigmoid')(y)
        
        return KerasModel(inputs=base_model.input,outputs=y)

