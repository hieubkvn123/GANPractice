import os
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

class ContractiveAutoencoder:
    def __init__(self, input_shape=(128, 128, 3), encoding_dim=32, lambda_=1e-4):
        self.input_shape = input_shape
        self.encoding_dim = encoding_dim
        self.lambda_ = lambda_

    ### MSE + Jacobian of encoder layer penalty ###
    def contractive_loss(self, y_true, y_pred):
        h = y_pred # self.model.get_layer('encoder').output
        W = self.model.get_layer('encoded').weights[0]

        # tf.print(h.shape, W.shape)

        contractive = (h * (1 - h)) @ K.transpose(W)

        loss = self.lambda_ * K.sum(contractive ** 2)

        # tf.print("Loss = " + str(loss))

        return loss

    def build(self):
        inputs = Input(shape=self.input_shape)

        ### Building the encoder ###
        ''' First conv block '''
        conv1 = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
        pool1 = MaxPooling2D((2,2))(conv1) ### Now shape = (64. 64. 64)
        norm1 = BatchNormalization()(pool1)

        ''' Second conv block '''
        conv2 = Conv2D(32, (3,3), activation='relu', padding='same')(norm1)
        pool2 = MaxPooling2D((2,2))(conv2) ### Now shape = (32, 32, 32)
        norm2 = BatchNormalization()(pool2)

        ''' Third conv block '''
        conv3 = Conv2D(16, (3,3), activation='relu', padding='same')(norm2)
        pool3 = MaxPooling2D((2,2))(conv3) ### Now shape = (16, 16, 16)
        norm3 = BatchNormalization()(pool3)

        ''' Final conv to get latent space representation '''
        ### Latent space representation shape = (8, 16, 16) ###
        conv4 = Conv2D(8, (3,3), activation='relu', padding='same')(norm3) 
       
        flatten = Flatten()(conv4)
        h = Dense(128, activation='sigmoid', name='encoded')(flatten)
        dense1 = Dense(8 * 16 * 16, activation='relu')(h)

        reshape = Reshape(target_shape=(16,16,8))(dense1)

        ### Building the decoder ###
        ''' First upsampling '''
        conv1 = Conv2D(64, (3,3), activation='relu', padding='same')(reshape)
        up1 = UpSampling2D((2,2))(conv1) ### shape = (64, 32, 32) ###

        ''' Second upsampling '''
        conv2 = Conv2D(32, (3,3), activation='relu', padding='same')(up1)
        up2 = UpSampling2D((2,2))(conv2) ### shape = (32, 64, 64) ###

        ''' Third upsampling '''
        conv3 = Conv2D(16, (3,3), activation='relu', padding='same')(up2)
        up3 = UpSampling2D((2,2))(conv3) ### shape = (16, 128, 128) ###

        ''' Fourth upsampling '''
        conv4 = Conv2D(16, (3,3), activation='relu', padding='same')(up3)
        up4 = UpSampling2D((2,2))(conv4) ### shape = (16, 256, 256) ###

        ''' Final conv to restore original image depth '''
        output = Conv2D(self.input_shape[-1], (3,3), activation='sigmoid', padding='same', name='decoded')(up4)

        ### We nned the weights of the model ###
        self.model = Model(inputs, [output, h])
        self.model.compile(optimizer='adam', loss=self.contractive_loss)

        return self.model
