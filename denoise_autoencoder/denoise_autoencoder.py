import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

class DenoiseAutoencoder():
    def __init__(self, input_shape=(128, 128, 1), encoding_dim=32, lambda_=1e-4):
        self.input_shape = input_shape
        self.filters = (16, 32, 64) 
        self.encoding_dim = encoding_dim

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
        inputs = Input(shape=self.input_shape)
        x = inputs

        for f in self.filters:
            x = Conv2D(f, kernel_size=(3,3), padding='same', strides=2)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization()(x)

        volumeSize = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(self.encoding_dimi, 
                kernel_regularizer=regularizers.l2(self.lambda_/2))(x)

        encoder = Model(inputs, latent, name='encoder')

        ### Build the decoder ###
        latentInputs = Input(shape=(self.encoding_dim,))
        x = Dense(np.prod(volumeSize[1:]), 
                kernel_regularizer=regularizers.l2(self.lambda_/2))(latentInputs)
        ### With this x becomes a 2D tensor ###
        x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

        for f in self.filters:
            x = Conv2DTranspose(f, kernel_size=(3,3), strides=2, padding='same')(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization()(x)

        ### A final Conv2DTranspose to get back the original channels ###
        x = Conv2DTranspose(self.input_shape[-1], kernel_size=(3,3), padding='same')(x)
        outputs = Activation("sigmoid")(x)

        decoder = Model(latentInputs, outputs, name='decoder')

        ### Build the autoencoder ###
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        adam = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=True)
        autoencoder.compile(optimizer=adam, loss='mse')

        return autoencoder, encoder, decoder
