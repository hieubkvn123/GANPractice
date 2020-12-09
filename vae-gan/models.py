import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers, optimizers

class VAE_GAN(object):
    def __init__(self, latent_dim=1024, input_shape=(128,128,3)):
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

        conv3 = Conv2D(256, kernel_size=(5,5), strides=2, padding='same')(relu2)
        norm3 = BatchNormalization()(conv3)
        relu3 = LeakyReLU()(norm3)

        flatten = Flatten()(relu3)
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
        dense = Dense(16*16*256)(inputs)
        norm  = BatchNormalization()(dense)
        relu  = LeakyReLU()(norm)

        reshape = Reshape(target_shape=(16, 16, 256))(relu)

        ### 32 x 32 x 256 ###
        conv1 = Conv2DTranspose(256, kernel_size=(5,5), strides=2, padding='same')(reshape)
        norm1 = BatchNormalization()(conv1)
        relu1 = LeakyReLU()(norm1)

        ### 64 x 64 x 128 ###
        conv2 = Conv2DTranspose(128, kernel_size=(5,5), strides=2, padding='same')(relu1)
        norm2 = BatchNormalization()(conv2)
        relu2 = LeakyReLU()(norm2)

        ### 128 x 128 x 32 ###
        conv3 = Conv2DTranspose(32, kernel_size=(5,5), strides=2, padding='same')(relu2)
        norm3 = BatchNormalization()(conv3)
        relu3 = LeakyReLU()(norm3)

        out = Conv2D(3, kernel_size=(5,5), padding='same', activation='tanh')(relu3)

        model = Model(inputs=inputs, outputs=out)
        return model

    def build_discriminator(self):
        inputs = Input(shape=self.input_shape)

        conv1 = Conv2D(32, kernel_size=(5,5), activation='relu')(inputs)

        conv2 = Conv2D(128, kernel_size=(5,5))(conv1)
        norm2 = BatchNormalization()(conv2)
        relu2 = LeakyReLU()(norm2)

        conv3 = Conv2D(256, kernel_size=(5,5))(relu2)
        norm3 = BatchNormalization()(conv3)
        relu3 = LeakyReLU()(norm3)

        l_tilde = Conv2D(256, kernel_size=(5,5))(relu3)
        norm4 = BatchNormalization()(l_tilde)
        relu4 = LeakyReLU()(norm4)

        dense = Dense(512)(relu4)
        norm5 = BatchNormalization()(dense)
        relu5 = LeakyReLU()(norm5)

        out = Dense(1)(relu5)

        model = Model(inputs=inputs, outputs=[out, l_tilde])
        return model

    def get_models(self):
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
