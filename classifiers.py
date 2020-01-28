# -*- coding:utf-8 -*-
import keras

from keras.models import Model as KerasModel
from keras.models import Sequential

from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Concatenate, LeakyReLU, GlobalMaxPooling2D, GlobalAveragePooling2D, Reshape
from keras.layers import TimeDistributed, LSTM

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
    
    def eval_generator(self, validation_generator, steps_per_epoch):
        scoreSeg,loss = self.model.evaluate_generator(validation_generator, steps=steps_per_epoch)
        return scoreSeg, loss

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
            y = Flatten()(x4)
        
        
        return KerasModel(inputs = x, outputs = y)

class Xception_main(Classifier):
#    Reference
#    https://www.groundai.com/project/faceforensics-learning-to-detect-manipulated-facial-images/1
#    https://medium.com/@gkadusumilli/image-recognition-using-pre-trained-xception-model-in-5-steps-96ac858f4206
#     learning-rate of 0.0002 and a batch-size of 32
    def __init__(self, learning_rate = 0.0001):
        self.model = self.init_model()
        self.based_model_last_block_layer_number = 132
        optimizer = Adam(lr = learning_rate)
        # optimizer = SGD(lr = 0.001)
        #        set trainable layer
        for layer in self.model.layers[:self.based_model_last_block_layer_number]:
            layer.trainable = True
        for layer in self.model.layers[self.based_model_last_block_layer_number:]:
            layer.trainable = True
        print(self.model.summary())
        
        self.model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        
    
    
    def init_model(self, include_top = True):
        img_width, img_height = 299, 299
        base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)
        
        # Top Model Block
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        y = Dense(1, activation='sigmoid', name='predictions')(x)
    
        return KerasModel(inputs=base_model.input,outputs=y)

class Xception_main_drop(Classifier):
#    Reference
#    https://www.groundai.com/project/faceforensics-learning-to-detect-manipulated-facial-images/1
#    https://medium.com/@gkadusumilli/image-recognition-using-pre-trained-xception-model-in-5-steps-96ac858f4206
#     learning-rate of 0.0002 and a batch-size of 32
    def __init__(self, learning_rate = 0.0001):
        self.model = self.init_model()
        self.based_model_last_block_layer_number = 132
        optimizer = Adam(lr = learning_rate)
        # optimizer = SGD(lr = 0.001)
        #        set trainable layer
        for layer in self.model.layers[:self.based_model_last_block_layer_number]:
            layer.trainable = False
        for layer in self.model.layers[self.based_model_last_block_layer_number:]:
            layer.trainable = True
        print(self.model.summary())
        
        self.model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        
    def init_model(self, include_top = True):
        img_width, img_height = 299, 299
        base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)
        
        # Top Model Block
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # y = Dense(1, activation='sigmoid', name='predictions')(x)
        
        y = Dropout(0.5)(x)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)
        
        return KerasModel(inputs=base_model.input,outputs=y)


# Reference:
#https://towardsdatascience.com/get-started-with-using-cnn-lstm-for-forecasting-6f0f4dde5826
#https://stackoverflow.com/questions/53488768/keras-functional-api-combine-cnn-model-with-a-rnn-to-to-look-at-sequences-of-im
#https://stackoverflow.com/questions/53488359/cnn-lstm-image-classification

class Meso_lstm(Classifier):
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
        optimizer = Adam(lr = learning_rate)
        self.model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    def init_model(self):
        
#        input_shape = (32, 5, 256,256,3), (batch, sequence, img_h, img_w, img_dimension)
#        out_timesDistributed_meso = (32, 5, 1024)
#        input_shape_lstm, as the same as above 
#        out_LSTM = (), 
#        return_sequences=True (32,5,32)
#        return_sequences=False (32,32), only the number in last times stamp left
        
        base_model_class = MesoInception4()
        base_model = base_model_class.init_model(include_top=False)
        for layer in base_model.layers:
            layer.trainable=False
        
#        return_sequences = False means
#       False in Keras RNN layers, and this means the RNN layer will only return the last hidden state output
        x = Input(shape = (5, IMGWIDTH, IMGWIDTH, 3))
        x1 = TimeDistributed(base_model)(x)
        x2 = LSTM(32, return_sequences=False)(x1)
        y = Dense(1, activation='sigmoid',use_bias=True)(x2)
        return KerasModel(inputs=x,outputs=y)
        
        
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

#        set trainable layer
        for layer in self.model.layers[:-2]:
            layer.trainable=False
        for layer in self.model.layers[-2:]:
            layer.trainable=True
            
        self.model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        
    def init_model(self, include_top = True):
        base_model = MobileNet(input_shape=(224,224,3), alpha=1.0, include_top=False, weights='imagenet',  pooling=False, classes=1)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        y = Dense(1, activation='sigmoid',use_bias=True)(x)
        
        return KerasModel(inputs=base_model.input,outputs=y)

