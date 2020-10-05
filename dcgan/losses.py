import os
import numpy as np
import tensorflow as tf

import tensorflow.keras.losses import BinaryCrossentropy as bce

'''
    This loss defines how well the discriminator can distinguish 
    from real and fake output by computing the crossentropy of the real output 
    with a tensor of ones and the crossentropy of the fake output with a 
    tensor of zeros
'''
def discriminator_loss(real_output, fake_output):
    real_loss = bce(tf.ones_like(real_output), real_output)
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)

    loss = real_loss + fake_loss

    return loss

'''
    This loss defines how well the generator is able to trick the discriminator
    here we will compare the fake output to a tensor of ones
'''
def generator_loss(fake_output):
    loss = bce(tf.ones_like(fake_output), fake_output)

    return loss

