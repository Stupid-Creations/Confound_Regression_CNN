from keras.models import *
from keras.layers import Activation, Dense, Flatten, Input, Layer, Conv3D, Reshape, BatchNormalization, Dropout, MaxPooling3D
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import nibabel
import csv
import os

def mse_inverse(y_true,y_pred):
  MSE = K.sum(K.square(y_pred-y_true))
  return -MSE

def read_scan(path):
  scan = nibabel.load(path)
  scanned = scan.get_fdata()
  return scanned

class triple_adverse():
  def __init__(self):
    self.image = Input(shape = (256,256,128,1), name = "input")

    self.feature = self.get_encoder()
    self.encoder = Model(self.image,self.feature)

    self.regressor = self.get_regressor()
    self.regressor.compile(loss = 'mse', optimizer = Adam(0.0002))
    self.regressor.trainable = False

    confound = self.regressor(self.feature)
    self.remover = Model(self.image,confound)
    self.remover.compile(loss = mse_inverse, optimizer = Adam(0.0002))

    self.classifier = self.get_classifier()

    self.modelmaker = self.classifier(self.feature)
    self.model = Model(self.image,self.modelmaker)
    self.model.compile(loss = 'binary_crossentropy',optimizer = Adam(0.0001))

  def get_encoder(self):
    print('making encoder')
    encoder = Conv3D(128,kernel_size = 3,strides = 2,activation = 'tanh',padding = 'same')
    print('a')
    encoder = BatchNormalization()(encoder)
    encoder = Conv3D(256,kernel_size = 3,strides = 2,activation = 'tanh',padding='same')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Conv3D(256,kernel_size = 3,strides = 2,activation = 'tanh',padding = 'same')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Conv3D(256,kernel_size = 3,strides = 2,activation = 'tanh',padding = 'same')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Flatten()(encoder)
    encoder = Dense(2048,activation = 'tanh')
    encoder = BatchNormalization()(encoder)
    encoder = Dense(2048,activation = 'tanh')
    encoder = BatchNormalization()(encoder)
    encoder = Dense(1024)(encoder)
    print('made encoder')
    return encoder
  def get_regressor(self):
    inputs_x = Input(shape=(1024,))
    feature = Dense(2048,activation = 'tanh')(inputs_x)
    feature = Dropout(0.3)(feature)
    feature = Dense(1024,activation = 'tanh')(feature)
    feature = Dropout(0.3)(feature)
    feature = Dense(1024,activation='tanh')(feature)
    feature = Dense(1024,activation='tanh')(feature)
    feature = Dense(3,activation = 'ReLU')(feature)
    regressor = Model(inputs = inputs_x,outputs=feature)
    return regressor
  def get_classifier(self):
    inputs_x = Input(shape=(1024,))
    feature = Dense(2048,activation = 'tanh')(inputs_x)
    feature = Dropout(0.3)(feature)
    feature = Dense(1024,activation = 'tanh')(feature)
    feature = Dropout(0.3)(feature)
    feature = Dense(1024,activation='tanh')(feature)
    feature = Dense(1024,activation='tanh')(feature)
    feature = Dense(1,activation = 'sigmoid')(feature)
    classifier = Model(inputs = inputs_x,outputs=feature)
    return classifier
  def train(self,clabel,input,tlabel,epochs):
    print('training')
    for i in range(epochs):
      for j in range(len(input)):
        encoded_data = self.encoder.predict(input[i])
        regression_data = tf.data.Dataset.from_tensor_slices((encoded_data,clabel[i]))
        self.regressor.fit(regression_data,epochs = 1)
        self.remover.fit(input[i],clabel[i])
        self.model.fit(input[i],tlabel[i])
      print("epoch"+str(i))
  def test_regressor(self,clabel,input):
        means = []
        for i in range(len(input)):
            encoded_data = self.encoder.predict(input[i])
            regression_data = tf.data.Dataset.from_tensor_slices((encoded_data,clabel[i]))
            regpredict = self.regressor.predict(encoded_data)
            MSE = -(mse_inverse(clabel,self.regpredict))
            means.append(MSE)
        return np.mean(MSE)
