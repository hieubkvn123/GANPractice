import os
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 

from argparse import ArgumentParser
from models import ConditionalGAN 
from losses import generator_loss, discriminator_loss
from tensorflow.keras.optimizers import Adam#, RMSProp
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K

parser = ArgumentParser()
parser.add_argument('--epochs', required=False, default=1000, help='Number of training iterations')
parser.add_argument('--batch_size', required=False, default=32, help='Number of images per batch')
parser.add_argument('--latent_dim', required=False, default=100, help='The dimension of the latent space')
parser.add_argument('--ckpt_dir', required=False, default='./checkpoints', help='Checkpoint directory')
args = vars(parser.parse_args())

### Some constants ###
EPOCHS = int(args['epochs'])
BATCH_SIZE = int(args['batch_size'])
LATENT_DIM = int(args['latent_dim'])
CHECKPOINT_DIR = args['ckpt_dir']

### Prepare the dataset ###
(X_train, Y_train), (_, _) = mnist.load_data()
num_classes = len(np.unique(Y_train))

X_train = (X_train - 127.5) / 127.5
X_train = X_train.reshape(-1, 28, 28, 1)
Y_train = tf.one_hot(Y_train, depth=num_classes).numpy().astype('uint8')

### Load the models ###
D, G = ConditionalGAN(latent_dim=LATENT_DIM).get_models()
print(G.summary())
print(D.summary())

### Create optimizers ###
g_opt, d_opt = Adam(1e-4), Adam(1e-4)

### Define training step ###
@tf.function 
def _train_step(images, labels):
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        z = K.random_normal(mean=0.0, stddev=1.0, shape=(K.shape(images)[0], LATENT_DIM))
        generated_img = G([z, labels], training=True)

        real_output = D(images, training=True)
        fake_output = D(generated_img, training=True)

        ### Compute losses ###
        g_loss = generator_loss(fake_output)
        d_loss = discriminator_loss(real_output, fake_output)

        ### Compute gradients of params w.r.t losses ###
        grad_g = g_tape.gradient(g_loss, G.trainable_variables)
        grad_d = d_tape.gradient(d_loss, D.trainable_variables)

        ### Backprobagation ###
        g_opt.apply_gradients(zip(grad_g, G.trainable_variables))
        d_opt.apply_gradients(zip(grad_d, D.trainable_variables))

    return K.mean(g_loss), K.mean(d_loss)

noise  = K.random_normal(mean=0.0, stddev=1.0, shape=(9, LATENT_DIM))
test_labels = np.array([1,2,3,4,5,6,7,8,9])
test_labels = tf.one_hot(test_labels, depth=num_classes)

def _generate_image(model, epoch, z, labels):
    '''
        Notice 'training' is set to False 
        This is to set all layers to run in inference mode
    '''

    predictions = model([z, labels], training=False)

    fig = plt.figure(figsize=(4,4))
    filename = 'images/image_at_epoch_{:04d}.png'.format(epoch)
    for i in range(predictions.shape[0]):
        plt.subplot(3,3, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(filename)

    ### Closes all created fig ###
    plt.close('all')

    return filename

def _generate_gif(image_list):
    with imageio.get_writer('images/output.gif', mode='I') as writer:
        for filename in image_list:
            image = imageio.imread(filename)
            writer.append_data(image)

### Main training function ###
def train(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE):
    gifs = []

    dataset_len = X.shape[0]
    steps_per_epoch = dataset_len // batch_size
    for i in range(epochs):
        for j in range(steps_per_epoch):
            batch_X = X[j * batch_size: (j+1) * batch_size]
            batch_Y = Y[j * batch_size: (j+1) * batch_size]

            g_loss, d_loss = _train_step(batch_X, batch_Y)
            print('[*] Epochs #[%d/%d]| Batch #[%d/%d]: Generator loss : %.4f - Discriminator loss : %.4f' % (
                i+1, 
                epochs,
                j+1,
                steps_per_epoch,
                g_loss, 
                d_loss
            ))

        ### produce image for gif ###
        filename = _generate_image(G, i+1, noise, test_labels)
        gifs.append(filename)

        if(i % 15 == 0):
            print('[*] Checkpointing generator and discriminator to %s' % CHECKPOINT_DIR)
            G.save_weights(os.path.join(CHECKPOINT_DIR, 'G.weights.hdf5'))
            D.save_weights(os.path.join(CHECKPOINT_DIR, 'D.weights.hdf5'))

    generate_gifs(gifs)

train(X_train, Y_train) 
