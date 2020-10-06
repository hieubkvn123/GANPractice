import os
import time
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
BATCH_SIZE=256
BUFFER_SIZE = 60000

### Create generator and discriminator ###
generator = make_generator()
discriminator = make_discriminator()

### Create optimizers for 2 networks ###
generator_optimizer = Adam(1e-4)
discriminator_optimizer = Adam(1e-4)

### Define the checkpoint directory ###
### We will checkpoint generator, discriminator and the optimizers ###
image_dir = './images'
checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator)

### Define the training step ###
'''
    Note the use of @tf.function
    This annotation causes the function to be compiled
'''
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    ### tf.GradientTape record operations for auto gradient ###
    ### just like mxnet autograd ###
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    g_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    g_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(g_generator,generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(g_discriminator, discriminator.trainable_variables))

'''
    Define 16 seed corresponding 16 images to generate for view
'''
seed = tf.random.normal([NUM_REAL, NOISE_DIM])

def generate_image(model, epoch, test_input):
    '''
        Notice 'training' is set to False 
        This is to set all layers to run in inference mode
    '''

    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4,4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(image_dir + '/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

### Prepare the dataset ###
(train_images, train_labels), (_,_)  = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
train_images = (train_images - 127.5)/127.5 ### Normalize image to [-1,1]

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        ### produce image for GIF as we go ###
        generate_image(generator, epoch + 1, seed)
        if ( epoch + 1 ) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    generate_image(generator, epochs, seed)

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train(train_dataset, EPOCHS)
