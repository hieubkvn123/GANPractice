import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

class VAE(object):
    def __init__(self, latent_dim=256, input_shape=(28,28,1)):
        self.latent_dim = latent_dim
        self.input_shape = input_shape

    def build(self):

