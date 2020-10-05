import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def make_generator(input_shape=(100,)):
    inputs = Input(shape=input_shape)
    x = Dense(7*7*256, use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Reshape((7, 7, 256))(x)
    x = Conv2DTranspose(128, kernel_size=(5,5), strides=(1,1), padding='same', use_bias=False)(x)
    ### x = 128 x 7 x 7 ###
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(64, kernel_size=(5,5), strides=(2,2), padding='same', use_bias=False)(x)
    ### x = 64 x 14 x 14 ###
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(1, kernel_size=(5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh')(x)
    ### x = 1 x 28 x 28 ###

    model = Model(inputs=inputs, outputs=x)

    return model

def make_discriminator(input_shape=(28, 28, 1)):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=(5,5), strides=(2,2), padding='same')(inputs)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, kernel_size=(5,5), strides=(2,2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=x)

    return model
