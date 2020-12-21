import os 
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class ConditionalGAN(object):
    def __init__(self, input_shape=(28,28,1), latent_dim=100, num_classes=10):
        self.input_shape = input_shape
        self.latent_dim  = latent_dim
        self.num_classes = num_classes

    def build_label_deconv(self):
        labels = Input(self.num_classes,))
        fc = Dense(7*7*16, activation='relu')(labels)

        reshape = Reshape(target_shape=(7,7,16))(fc)

        ### 14 x 14 x 32 ###
        deconv1 = Conv2DTranspose(32, kernel_size=(5,5), strides=(2,2), padding='same')(reshape)
        btnorm1 = BatchNormalization()(deconv1)
        relu1   = LeakyReLU(alpha=0.2)(btnorm1)

        model = Model(inputs=labels, outputs=relu1)

        return model

    def build_generator(self):
        inputs = Input(shape=(self.latent_dim, ))
        labels = Input(shape=(self.num_classes,))

        fc = Dense(7*7*16, activation='relu')(inputs)
        reshape = Reshape(target_shape=(7,7,16))(fc)

        ### 14 x 14 x 32 ###
        deconv1 = Conv2DTranspose(32, kernel_size=(5,5), strides=(2,2), padding='same')(reshape)
        btnorm1 = BatchNormalization()(deconv1)
        relu1   = LeakyReLU(alpha=0.2)(btnorm1)

        ### Merge with label's representation ###
        y = self.build_label_deconv()(labels)
        merged = Concatenate(axis=-1)([relu1, y]) ### 14 x 14 x 64 ###

        ### 28 x 28 x 64 ###
        deconv2 = Conv2DTranspose(64, kernel_size=(5,5), strides=(2,2), padding='same')(merged)
        btnorm2 = BatchNormalization()(deconv2)
        relu2   = LeakyReLU(alpha=0.2)(btnorm2)

        out = Conv2D(1, kernel_size=(5,5), padding='same', strides=(1,1))(relu2)

        generator = Model(inputs=[inputs, labels], outputs=out)
        return generator

    def build_discriminator(self):
        inputs = Input(shape=self.input_shape)

        x = Conv2D(64, kernel_size=(5,5), strides=(2,2), padding='same')(inputs)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)

        x = Conv2D(128, kernel_size=(5,5), strides=(2,2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)

        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)

        discriminator = Model(inputs=inputs, outputs=x)
        return discriminator
