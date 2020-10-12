import os
import cv2
import time
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from tensorflow.keras.optimizers import Adam

from models import make_generator, make_discriminator
from losses import generator_loss, discriminator_loss
from data_loader import data_from_dir

parser = ArgumentParser()
parser.add_argument('-d', '--data_dir', required=True, help='Path to data directory')
parser.add_argument('--num_images', required=False, help='Number of training images')
parser.add_argument('--batch_size', required=False, help='Number of images per batch', default=256)
parser.add_argument('--noise_dim', required=False, help='Dimension of the noise vector', default=100)
parser.add_argument('--epochs', required=False, help='Number of training iterations', default=50)

args = vars(parser.parse_args())
EPOCHS = int(args['epochs'])
NOISE_DIM = int(args['noise_dim'])
NUM_REAL = 16
NUM_IMAGES = int(args['num_images'])
BATCH_SIZE = int(args['batch_size'])

### Load images from folder ###
images = data_from_dir(args['data_dir'], max_num_img=NUM_IMAGES)

### Create models and optimizers for checkpoints ###
generator = make_generator()
discriminator = make_discriminator()
generator_opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, amsgrad=True)
discriminator_opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, amsgrad=True)

### Create checkpoint ###
image_dir = './images'
checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(generator=generator,
        discriminator=discriminator,
        generator_opt=generator_opt,
        discriminator_opt=discriminator_opt)

### Check if the checkpoint directory is empty ###
if(len(os.listdir(checkpoint_dir)) != 0):
    print('[*] Restoring latest checkpoint ... ')
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
else:
    print('[*] No checkpoint found ... ')

### Define the training steps ###
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    loss = None

    ### tf.GradientTape records operations for automatic differentiation ###
    ### just like autograd in mxnet ###
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        loss = gen_loss + disc_loss
        loss = tf.math.reduce_mean(loss)

    g_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    g_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_opt.apply_gradients(zip(g_generator, generator.trainable_variables))
    discriminator_opt.apply_gradients(zip(g_discriminator, discriminator.trainable_variables))

    return loss

seed = tf.random.normal([NUM_REAL, NOISE_DIM])

def generate_image(generator, epoch, seed):
    predictions = generator(seed, training=False)

    fig = plt.figure(figsize=(8,8))
    filename = image_dir + "/image_at_epoch_{:04d}.png".format(epoch)
    for i in range(predictions.shape[0]):
        plt.subplot(4,4, i+1)
        plt.imshow(predictions[i,:,:,0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")

    plt.savefig(filename)

    ### Close all figs ###
    plt.close("all")

    return filename

def generate_gif(image_list):
    with imageio.get_writer('images/output.gif', mode='I') as writer:
        for filename in image_list:
            image = imageio.imread(filename)
            writer.append_data(image)

def train(images, epochs):
    gifs = [] ### list of images to create a gif ###

    ### Delete old gifs and images ###
    if(os.path.exists(image_dir + "/*.gif")) : os.remove(image_dir + "/*.gif")
    if(os.path.exists(image_dir + "/*.png")) : os.remove(image_dir + "/*.png")

    loss = 0
    for epoch in range(epochs):
        start = time.time()
        num_batches = images.shape[0] // BATCH_SIZE

        print('[*] Epoch %4d : ' % (epoch + 1))
        for i in range(num_batches):
            image_batch = images[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            loss = train_step(image_batch)
            print('   ---> Finished processing batch %3d, loss : %.4f' % (i+1, loss))
        print('[*] Time taken : %.2f seconds' % (time.time() - start))

        ### produce image for GIF as we go ###
        file_name = generate_image(generator, epoch + 1, seed)
        gifs.append(file_name)

        ### Checkpoint every 15 epochs ###
        if((epoch + 1) % 15 == 0):
            print('[*] Saving models weights and making checkpoints ... ')
            checkpoint.save(file_prefix=checkpoint_prefix)
            generator.save_weights('weights/generator.weights/hdf5')
            discriminator.save_weights('weights/discriminator.weights.hdf5')

    generate_gif(gifs)

### Normalize images to [-1, 1] ###
images = (images - 127.5) / 127.5
train(images, EPOCHS)
