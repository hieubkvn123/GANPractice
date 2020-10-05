import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from models import make_generator, make_discriminator
from losses import generator_loss, discriminator_loss

### Some constants ###
EPOCHS=50
NOISE_DIM=100
NUM_REAL=16 # number of real samples to generate

### Create generator and discriminator ###
generator = make_generator()
discriminator = make_discriminator()

### Create optimizers for 2 networks ###
generator_optimizer = Adam(1e-4)
discriminator_optimizer = Adam(1e-4)

### Define the checkpoint directory ###
### We will checkpoint generator, discriminator and the optimizers ###
checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator)


