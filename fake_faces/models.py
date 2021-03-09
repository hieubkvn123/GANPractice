import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

### Generator (The artist) ###
### Desired output = 128 x 128 x 1 ###
def make_generator(input_shape=(100,)):
    inputs = Input(shape=input_shape)
    x = Dense(32 * 32 * 256, use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Reshape((32, 32, 256))(x)
    ### x = 32 x 32 x 128 ###
    x = Conv2DTranspose(128, kernel_size=(5,5), strides=(1,1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    ### x = 64 x 64 x 64 ###
    x = Conv2DTranspose(64, kernel_size=(5,5), strides=(2,2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    ### x = 64 x 64 x 64 ###
    x = Conv2DTranspose(32, kernel_size=(5,5), strides=(2,2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    ### x = 128 x 128 x 1 ###
    x = Conv2DTranspose(3, kernel_size=(5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh')(x)

    model = Model(inputs, x)
    return model

### Discriminator (The art critic) ###

def make_discriminator(input_shape=(256,256,3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=(5,5), strides=(2,2), padding='same')(inputs)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, kernel_size=(5,5), strides=(2,2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, kernel_size=(5,5), strides=(2,2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    
    x = Flatten()(x)
    x = Dense(1)(x)

    model = Model(inputs, x)
    return model
