import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers, optimizers

class VAE_GAN(object):
    def __init__(self, latent_dim=2, input_shape=( 28, 28,3)):
        self.latent_dim = latent_dim
        self.input_shape = input_shape

    def sampling(self, x):
        mu, logvar = x
        sigma = Lambda(lambda x : K.exp(0.5 * x))(logvar)

        ### Reparameterize : z = mu + sigma * epsilon ###
        epsilon = K.random_normal(mean=0, stddev=1, shape=(K.shape(mu)[0], self.latent_dim))
        z = Multiply()([sigma, epsilon])
        z = Add()([z, mu])

        return z

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        conv1 = Conv2D(64, kernel_size=(5,5), strides=2, padding='same')(inputs)
        norm1 = BatchNormalization()(conv1)
        relu1 = LeakyReLU()(norm1)

        conv2 = Conv2D(128, kernel_size=(5,5), strides=2, padding='same')(relu1)
        norm2 = BatchNormalization()(conv2)
        relu2 = LeakyReLU()(norm2)

        flatten = Flatten()(relu2)
        dropout = Dropout(0.5)(flatten)

        mu = Dense(self.latent_dim)(dropout)
        mu = BatchNormalization()(mu)
        mu = LeakyReLU()(mu)

        logvar = Dense(self.latent_dim)(dropout)
        logvar = BatchNormalization()(logvar)
        logvar = LeakyReLU()(logvar)

        z = Lambda(lambda x : self.sampling(x))([mu, logvar])

        model = Model(inputs=inputs, outputs=[mu, logvar, z], name='enc')
        return model

    def build_decoder(self):
        inputs = Input(shape=(self.latent_dim,))
        dense = Dense(7*7*128)(inputs)
        norm  = BatchNormalization()(dense)
        relu  = LeakyReLU()(norm)

        reshape = Reshape(target_shape=(7, 7, 128))(relu)

        ### 7 x 7 x 256 ###
        conv1 = Conv2DTranspose(256, kernel_size=(5,5), strides=(1,1), use_bias=False, padding='same')(reshape)
        norm1 = BatchNormalization()(conv1)
        relu1 = LeakyReLU()(norm1)

        ### 14 x 14 x 128 ###
        conv2 = Conv2DTranspose(128, kernel_size=(5,5), strides=(2,2), use_bias=False, padding='same')(relu1)
        norm2 = BatchNormalization()(conv2)
        relu2 = LeakyReLU()(norm2)

        ### 28 x 28 x 3 ###
        out = Conv2DTranspose(3, kernel_size=(5,5), strides=(2,2), use_bias=False, padding='same', activation='tanh')(relu2)

        model = Model(inputs=inputs, outputs=out, name='dec')
        return model

    def build_discriminator(self):
        inputs = Input(shape=self.input_shape)
        x = Conv2D(64, kernel_size=(5,5), strides=(2,2), padding='same')(inputs)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)

        l_tilde = Conv2D(128, kernel_size=(5,5), strides=(2,2), padding='same')(x)
        x = LeakyReLU()(l_tilde)
        x = Dropout(0.3)(x)

        x = Flatten()(x)
        x = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=[x, l_tilde])
        return model 

    def get_models(self):
        print(self.enc.summary())
        print(self.dec.summary())
        print(self.dis.summary())

        return self.enc, self.dec, self.dis

    def build(self):
        inputs = Input(shape=self.input_shape)
        self.enc = self.build_encoder()
        self.dec = self.build_decoder()
        self.dis = self.build_discriminator()

        ### Output as a decoder ###
        mu, logvar, z_tilde = self.enc(inputs)
        x_tilde = self.dec(z_tilde)

        ### Output as a generator ###
        z_p = K.random_normal(mean=0, stddev=1.0, shape=(K.shape(mu)[0], self.latent_dim))
        x_p = self.dec(z_p)

        d_real, l_real = self.dis(inputs)
        d_fake, l_fake = self.dis(x_p)
        d_tilde, l_tilde = self.dis(x_tilde)

        vaegan = Model(inputs=inputs, outputs=[l_tilde, l_real, mu, logvar, d_real, d_fake, d_tilde])
        
        return vaegan

