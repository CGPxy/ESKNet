# coding=utf-8
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from PIL import Image
import cv2
import random
import os
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.python.layers import utils
from tensorflow.keras import regularizers
from model_function import *
from tensorflow.keras import layers

img_w = 384  
img_h = 384


def soout(data, size,name):
    outconv = Conv2D(1, (1, 1), strides=(1, 1), padding='same')(data)
    out = Activation('sigmoid')(outconv)
    up = UpSampling2D(size=size,name=name)(out)

    return up

def SKConvBlock0(data, filte):
    conv1 = Conv2D(filte, (3, 3), padding="same")(data)
    batch1 = BatchNormalization()(conv1)
    LeakyReLU1 = LeakyReLU(alpha=0.01)(batch1) 

    return LeakyReLU1

def SKConvBlock1plus(data, filte):
    conv2 = Conv2D(filte, (3, 3), padding="same", dilation_rate=(3,3))(data)
    batch2 = BatchNormalization()(conv2)
    LeakyReLU2 = LeakyReLU(alpha=0.01)(batch2)

    conv4 = Conv2D(filte, (5, 5), padding="same")(data)
    batch4 = BatchNormalization()(conv4)
    LeakyReLU4 = LeakyReLU(alpha=0.01)(batch4)

    condata = Concatenate()([LeakyReLU4, LeakyReLU2])

    return condata, LeakyReLU2, LeakyReLU4

def SKConvnblockplus(data, filte, size):
    data1 = SKConvBlock0(data=data, filte=filte)
    data2, data21, data22 = SKConvBlock1plus(data=data1, filte=filte)

    # Channel block
    data3 = GlobalAveragePooling2D()(data2)
    data3 = Dense(units=filte)(data3)
    data3 = BatchNormalization()(data3)
    data3 = ReLU()(data3)
    data3 = Dense(units=filte)(data3)
    data3 = Activation('sigmoid')(data3)

    a = Reshape((1, 1, filte))(data3)
    data_a = multiply([data21, a])

    a1 = 1-data3
    a1 = Reshape((1, 1, filte))(a1)
    data_a1 = multiply([data22, a1])

    # spatial block
    conv5 = ReLU()(data2)
    conv5 = Conv2D(1, (1,1), padding="same")(conv5)
    conv5 = Activation('sigmoid')(conv5)
    data5 = expend_as(conv5, filte)
    data5 = multiply([data5, data21])

    a2 = 1-conv5
    a2 = expend_as(a2, filte)
    data_a2 = multiply([data22, a2])

    data_a_a1 = Concatenate()([data_a, data_a1, data_a2, data5, data1])
    data_a_a1 = Conv2D(filte, (1,1), padding="same")(data_a_a1) #?????
    data_a_a1 = BatchNormalization()(data_a_a1)
    data_a_a1 = LeakyReLU(alpha=0.01)(data_a_a1) #?????
    return data_a_a1

def SKupdataplus(filte, data, skipdata, size):
    data1 = UpSampling2D((2, 2))(data)
    skipdata0 = Concatenate()([skipdata,data1])

    conv1 = SKConvnblockplus(data=skipdata0, filte=filte, size=size)
    conv2 = SKConvnblockplus(data=conv1, filte=filte, size=size)
    return conv2

def ESKNet():
    inputs = Input((img_h, img_w, 3))
    Conv1 = SKConvnblockplus(data=inputs, filte=32,size=(1,1))
    Conv1 = SKConvnblockplus(data=Conv1, filte=32,size=(1,1))

    pool1 = MaxPooling2D(pool_size=(2, 2))(Conv1)
    Conv2 = SKConvnblockplus(data=pool1, filte=64, size=(3,3))
    Conv2 = SKConvnblockplus(data=Conv2, filte=64, size=(3,3)) 

    pool2 = MaxPooling2D(pool_size=(2, 2))(Conv2)
    Conv3 = SKConvnblockplus(data=pool2, filte=128, size=(1,1))
    Conv3 = SKConvnblockplus(data=Conv3, filte=128, size=(1,1))

    pool3 = MaxPooling2D(pool_size=(2, 2))(Conv3)   
    Conv4 = SKConvnblockplus(data=pool3, filte=256, size=(3,3))
    Conv4 = SKConvnblockplus(data=Conv4, filte=256, size=(3,3))

    pool4 = MaxPooling2D(pool_size=(2, 2))(Conv4)    
    Conv5 = SKConvnblockplus(data=pool4, filte=512, size=(1,1))
    Conv5 = SKConvnblockplus(data=Conv5, filte=512, size=(1,1))
    out5 = soout(data=Conv5, size=(16,16),name='out5')

    # 48
    up1 = SKupdataplus(filte=256, data=Conv5, skipdata=Conv4, size=(3,3))
    out4 = soout(data=up1, size=(8,8),name='out4')
    # 96
    up2 = SKupdataplus(filte=128, data=up1, skipdata=Conv3, size=(1,1))
    out3 = soout(data=up2, size=(4,4),name='out3')
    # 192
    up3 = SKupdataplus(filte=64, data=up2, skipdata=Conv2, size=(3,3))
    out2 = soout(data=up3, size=(2,2),name='out2')
    
    # 384
    up4 = SKupdataplus(filte=32, data=up3, skipdata=Conv1, size=(1,1))
    outconv = Conv2D(1, (1, 1), strides=(1, 1), padding='same')(up4)
    out1 = Activation('sigmoid',name='out1')(outconv)

    model = Model(inputs=inputs, outputs=[out1,out2,out3,out4,out5])
    return model
