import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

class SparseAutoencoder():
    def __init__(self, input_shape=(28,28,1),lambda_=0.001, sparsity=0.01, beta=3, encoding_dim=16):
        self.input_shape = input_shape
        self.encoding_dim = encoding_dim

        self.p = sparsity
        self.lambda_ = lambda_
        self.beta = beta
        self.filters = (32, 64)

    ### Take the activation of the hidden layer ###
    ### Then regularize it using Kullback Leiber Divergence with the average of the batch sample
    ### and the sparsity parameter ###
    def kld_regularizer(self, activated_mat):
        p_hat = K.mean(activated_mat)
        KLD = self.p * K.log(self.p/p_hat) + (1 - self.p) * K.log((1-self.p) / (1 - p_hat))

        return self.beta * K.sum(KLD)


    def build(self):
        ### Building the encoder ###
        inputs = Input(shape=self.input_shape)
        x = inputs 

        for f in self.filters:
            x = Conv2D(f, kernel_size=(3,3), strides=2, padding='same')(x)
            x = LeakyReLU(alpha=0.2)(x)

            ### Normalize along the channels axis ###
            x = BatchNormalization(axis=-1)(x)

        volumeSize = K.int_shape(x) ### Get shape to reshape later in decoder ###
        x = Flatten()(x)
        latent = Dense(self.encoding_dim, 
                kernel_regularizer=regularizers.l2(self.lambda_/2))(x)

        encoder = Model(inputs, latent, name='encoder')

        ### Building the decoder ###
        latentInputs = Input(shape=(self.encoding_dim,))
        x = Dense(np.prod(volumeSize[1:]),
                kernel_regularizer=regularizers.l2(self.lambda_/2))(latentInputs)
        x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

        for f in self.filters[::-1]:
            x = Conv2DTranspose(f, kernel_size=(3,3), strides=2, padding='same')(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=-1)(x)

        x = Conv2DTranspose(self.input_shape[-1], kernel_size=(3,3), padding='same')(x)
        outputs = Activation("sigmoid")(x)

        decoder = Model(latentInputs, outputs, name='decoder')

        ### Building the autoencoder ###
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')

        adam = optimizers.Adam(lr=1e-4)
        autoencoder.compile(optimizer=adam, loss='mse')

        return autoencoder, encoder, decoder

