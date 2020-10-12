import tensorflow as tf

from tensorflow.keras.losses import BinaryCrossentropy

bce = BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    ones = tf.ones_like(fake_output)
    loss = bce(ones, fake_output)

    return loss

def discriminator_loss(real_output, fake_output):
    ones = tf.ones_like(real_output)
    zeros = tf.zeros_like(fake_output)

    real_loss = bce(ones, real_output)
    fake_loss = bce(zeros, fake_output)

    loss = real_loss + fake_loss

    return loss
