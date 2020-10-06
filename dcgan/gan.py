import os
import time
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

### Check if the checkpoint dir is empty, if not restore checkpoint ###
if(len(os.listdir(checkpoint_dir)) != 0):
    print('[*] Restoring last checkpoint ...')
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
else:
    print('[*] No checkpoints found ...')

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
    filename = image_dir + '/image_at_epoch_{:04d}.png'.format(epoch)
    for i in range(predictions.shape[0]):
        plt.subplot(4,4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(filename)

    ### Closes all created fig ###
    plt.close('all')

    return filename

def generate_gif(image_list):
    with imageio.get_writer('images/output.gif', mode='I') as writer:
        for filename in image_list:
            image = imageio.imread(filename)
            writer.append_data(image)

### Prepare the dataset ###
(train_images, train_labels), (_,_)  = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
train_images = (train_images - 127.5)/127.5 ### Normalize image to [-1,1]

def train(dataset, epochs):
    gifs = [] ### list of image to form gif ###

    if(os.path.exists(image_dir+"/*.gif")):os.remove(image_dir + "/*.gif") ### delete old gif ###
    if(os.path.exists(image_dir+"/*.png")):os.remove(image_dir + "/*.png")
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        ### produce image for GIF as we go ###
        filename = generate_image(generator, epoch + 1, seed)
        gifs.append(filename)
        if ( epoch + 1 ) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    generate_gif(gifs)

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train(train_dataset, EPOCHS)
