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

    def build_generator(self):
        inputs = Input(shape=(self.latent_dim,))
        label_inputs = Input(shape=(self.num_classes,)) ### The onehot encoded label ###

        concat = Concatenate(axis=-1)([inputs, label_inputs])


