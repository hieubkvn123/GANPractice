import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

class SparseAutoencoder():
    def __init__(self, input_shape=(28,28,1),lambda=0.001, sparsity=0.01, beta=3, encoding_dim=200):
        self.input_shape = input_shape
        self.p = sparsity
        self.lambda = lambda
        self.beta = beta

    ### Take the activation of the hidden layer ###
    ### Then regularize it using Kullback Leiber Divergence with the average of the batch sample
    ### and the sparsity parameter ###
    def kld_regularizer(self, activated_mat):
        p_hat = K.mean(activated_mat)
        KLD = p * K.log(p/p_hat) + (1 - p) * K.log((1-p) / (1 - p_hat))

        return self.beta * K.sum(KLD)

    def get_model(self):
        inputs = Input(shape=self.input_shape)
        encoded = Dense(self.encoding_dim, activation='sigmoid',
                activity_regularizer=self.kld_regularizer, ### Applied on the activation ###
                kernel_regularizer=regularizers.l2(self.lambda/2)(inputs) ### Applied on the parameters ###

        decoded = Dense(self.input_shape[0] * self.input_shape[1], activation='sigmoid',
                activity_regularizer=self.kld_regularizer, 
                kernel_regularizer=regularizers.l2(self.lambda/2)(encoded)


        autoencoder = Model(inputs, decoded)
        encoder = Model(inputs, encoded)
        
        encoded_inputs = Input(shape=(self.encoding_dim,))
        decoder_output = autoencoder.layers[-1](encoded_inputs)
        decoder = Model(encoded_inputs, decoder_output)

        adam = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=True)
        autoencoder.compile(optimizer=adam, loss='mse')

        return autoencoder, encoder, decoder
