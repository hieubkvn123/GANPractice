import numpy as np
import tensorflow as tf

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import backend as K

bce = BinaryCrossentropy(from_logits=True)
def generator_loss(fake_output):
    ones = tf.ones_like(fake_output)
    loss = bce(ones, fake_output)
    loss = tf.math.reduce_sum(loss)

    return loss

def discriminator_loss(real_output, fake_output):
    ones = tf.ones_like(real_output)
    zeros = tf.zeros_like(real_output)

    real_loss = bce(ones, real_output)
    fake_loss = bce(zeros, fake_output)
    loss = real_loss + fake_loss
    loss = tf.math.reduce_sum(loss)

    return loss
