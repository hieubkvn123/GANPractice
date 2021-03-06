import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
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
        ### Build the encoder ###
        x = Input(shape=self.input_shape)
        
        # Encoder
        e_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', 
                kernel_regularizer=l2(self.lambda_/2))(x)
        pool1 = MaxPooling2D((2, 2), padding='same')(e_conv1)
        batchnorm_1 = BatchNormalization()(pool1)
        
        e_conv2 = Conv2D(32, (3, 3), activation='relu', padding='same',
                kernel_regularizer=l2(self.lambda_/2))(batchnorm_1)
        h = MaxPooling2D((2, 2), padding='same')(e_conv2)


        # Decoder
        d_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same',
                kernel_regularizer=l2(self.lambda_/2))(h)
        up1 = UpSampling2D((2, 2))(d_conv1)
        
        d_conv2 = Conv2D(32, (3, 3), activation='relu', padding='same',
                kernel_regularizer=l2(self.lambda_/2))(up1)
        up2 = UpSampling2D((2, 2))(d_conv2)
        
        ### One last Conv to restore original depth of image ###
        r = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up2)

        model = Model(x, r)
        model.compile(optimizer='adam', loss='mse')

        return model #autoencoder, encoder, decoder
