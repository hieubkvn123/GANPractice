import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from models import make_generator, make_discriminator

### Some constants ###
NUM_TEST=16
NOISE_DIM=100

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

if(len(os.listdir(checkpoint_dir)) == 0):
    print('[*] Checkpoint does not exist ...')
else:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

generator=checkpoint.generator

print('[*] Generate image (y/n) : ')
choice = input()
choice.lower()

while(choice == 'y'):
    test_data = tf.random.normal([NUM_TEST, NOISE_DIM])

    generated_img = generator(test_data, training=False)
    generated_img = generated_img * 127.5 + 127.5

    fig, ax = plt.subplots(4,4, figsize=(4,4))
    for i in range(generated_img.shape[0]):
        img = generated_img[i, :, :, 0]
        row_id = i // 4
        col_id = i % 4

        ax[row_id][col_id].imshow(img, cmap='gray')
        plt.axis('off')

    plt.show()

    print('[*] Generate image (y/n) : ')
    choice = input()
    choice.lower()
