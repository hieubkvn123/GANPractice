import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

class DenoiseAutoencoder():
    def __init__(self, input_shape=(128, 128, 3), encoding_dim=32, lambda_=1e-4):
        self.input_shape = input_shape
        self.filters = (4, 8, 16, 32, 64) 
        self.encoding_dim = encoding_dim
        self.lambda_ = lambda_

    def awgn(self, image):
        ### Create random gaussian noise matrix ###
        col, row, ch = image.shape
        mean = 0
        sigma = 5 ### The more noise, the greater the sigma ###
        noise = np.random.normal(mean, sigma, (col, row, ch))
        noisy = image + noise
        noisy = noisy.astype(np.uint8)

        return noisy


    def build(self):
        ### Build the encoder ###
        x = Input(shape=self.input_shape)
        
        # Encoder
        e_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        pool1 = MaxPooling2D((2, 2), padding='same')(e_conv1)
        batchnorm_1 = BatchNormalization()(pool1)
        e_conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(batchnorm_1)
        pool2 = MaxPooling2D((2, 2), padding='same')(e_conv2)
        batchnorm_2 = BatchNormalization()(pool2)
        e_conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(batchnorm_2)
        h = MaxPooling2D((2, 2), padding='same')(e_conv3)


        # Decoder
        d_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(h)
        up1 = UpSampling2D((2, 2))(d_conv1)
        d_conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
        up2 = UpSampling2D((2, 2))(d_conv2)
        d_conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(up2)
        up3 = UpSampling2D((2, 2))(d_conv3)
        r = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up3)

        model = Model(x, r)
        model.compile(optimizer='adam', loss='mse')

        return model #autoencoder, encoder, decoder
