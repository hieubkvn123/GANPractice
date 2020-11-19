import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

class VAE(object):
    def __init__(self, latent_dim=2, input_shape=(28,28,1)):
        self.latent_dim = latent_dim
        self.input_shape = input_shape

    def decoder(self, name='decoder'):
        decoder_input = Input(shape=(self.latent_dim,))
        dense1 = Dense(7 * 7 * 16, activation='relu', name='decoder_input')(decoder_input)
        reshape = Reshape(target_shape=(7,7,16))(dense1)
        
        ### Now tensor size = 14 x 14 x 16 ###
        conv1 = Conv2D(16, kernel_size=(3,3), activation='relu', padding='same')(reshape)
        up1 = UpSampling2D((2,2))(conv1)

        ### Now tensor size = 28 x 28 x 8 ###
        conv2 = Conv2D(8, kernel_size=(3,3), activation='relu', padding='same')(up1)
        up2 = UpSampling2D((2,2))(conv2)

        decoder_output = Conv2D(1, kernel_size=(3,3), padding='same')(up2)
        decoder = Model(decoder_input, decoder_output, name=name)

        return decoder 

    def build(self):
        inputs = Input(shape=self.input_shape)
        
        ### Now tensor size = 14 x 14 x 8 ###
        conv1 = Conv2D(8, kernel_size=(3,3),  activation='relu', padding='same')(inputs)
        norm1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2,2))(norm1)

        ### Now tensor size = 7 x 7 x 16 ###
        conv2 = Conv2D(16, kernel_size=(3,3), activation='relu', padding='same')(pool1)
        norm2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2,2))(norm2)

        flatten = Flatten()(pool2)
        z_mu = Dense(self.latent_dim, name='encoded_mean')(flatten)
        z_log_var = Dense(self.latent_dim, name='encoded_logvar')(flatten)
        z_sigma = Lambda(lambda x : K.exp(x) ** 0.5 )(z_log_var)

        ### Reparameterize : z = mu + sigma * epsilon ###
        ### eps ~ N(1, 0) ###
        z = K.random_normal(stddev=1, mean=0, shape=(K.shape(z_mu)[0], self.latent_dim))
        z = Multiply()([z_sigma, z])
        z = Add(name='latent_input')([z, z_mu])

        ### Build the decoder ###
        decoder = self.decoder(name='decoder')
        output  = decoder(z)

        ### Rename output tensor ###
        decoded_output = Lambda(lambda x : x, name='decoded_output')(output) 
        encoded_output = Concatenate(axis=1, name='encoded_output')([z_mu, z_log_var])

        model = Model(inputs=inputs, outputs=[decoded_output, encoded_output]) # [z_mu, z_log_var, output])
        adam = optimizers.Adam(lr=1e-3, amsgrad=True)

        losses = {
                'decoded_output' : tf.keras.losses.mean_squared_error,
                'encoded_output' : self.kl_divergence_loss
        }
        model.compile(optimizer=adam, loss=losses)

        return model, decoder

    def kl_divergence_loss(self, y_true, y_pred):
        mu, log_var = tf.split(y_pred, num_or_size_splits=2, axis=1)
        print(mu.shape, log_var.shape)

        ''' The formula is calculate KL divergence of two gaussian distributions '''
        kl_divergence = -0.5 * tf.reduce_sum(1 + log_var -
                                    K.square(mu) - 
                                    K.exp(log_var), axis=-1)

        kl_divergence = tf.reduce_mean(kl_divergence)

        return kl_divergence 






