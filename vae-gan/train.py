import os
import imageio
import numpy as np
import tensorflow as tf

from models import VAE_GAN
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from dataset import write_to_tfrecord, read_from_tfrecord 

### Some constants ###
latent_dim = 1024
data_dir = '../../datasets/CASIA-WebFace'
record_file = 'data/casia-webface_1000.tfrecord'

### Initialize models ###
vae_gan = VAE_GAN()
vae_gan.build()

bce = BinaryCrossentropy(from_logits=True)
enc, dec, dis = vae_gan.get_models()
enc_opt, dec_opt, dis_opt = Adam(1e-4), Adam(1e-4), Adam(1e-4)

def kl_divergence(mu, logvar):
    ### Calculated from KL(N(mu, sigma), N(0,I)) ###
    kl_batch = -0.5 * K.sum(1 + logvar - \
                            K.square(mu) - \
                            K.exp(logvar), axis=-1)
    loss = K.mean(kl_batch)

    return loss

def reconstruction_loss(l_tilde, l_real):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(l_tilde - l_real), axis=[1,2,3]))
    return loss

def d_loss(d_real, d_fake, d_tilde):
    eps = 1e-8
    # loss = -tf.reduce_mean(tf.math.log(d_real+eps) + tf.math.log(1.0-d_fake+eps) + tf.math.log(1.0-d_tilde+eps))
    ones = tf.ones_like(d_real)
    zeros = tf.zeros_like(d_real)

    real_loss = bce(ones, d_real + eps)
    fake_loss = bce(zeros, d_fake + eps)
    tilde_loss = bce(zeros, d_tilde + eps)
    loss = real_loss + fake_loss + tilde_loss
    loss = K.mean(loss)

    return loss

def g_loss(d_fake, d_tilde):
    eps = 1e-8
    # loss = -tf.reduce_mean(tf.math.log(d_tilde+eps) + tf.math.log(d_fake+eps))
    ones = tf.ones_like(d_fake)
    fake_loss = bce(ones, d_fake + eps)
    tilde_loss = bce(ones, d_tilde + eps)
    loss = fake_loss + tilde_loss 
    loss = K.mean(loss)

    return loss

def _generate_gif(image_list):
    with imageio.get_writer('output/output.gif', mode='I') as writer:
        for image in image_list:
            image = imageio.core.util.Array(image)
            writer.append_data(image)

@tf.function
def _train_step(images):
    with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as dis_tape:
        ### Output as a decoder ###
        mu, logvar, z_tilde = enc(images, training=True)
        x_tilde = dec(z_tilde, training=True)

        ### Output as a generator ###
        z_p = K.random_normal(mean=0, stddev=1.0, shape=(K.shape(mu)[0], latent_dim))
        x_p = dec(z_p, training=True)

        d_real, l_real = dis(images, training=True)
        d_fake, l_fake = dis(x_p, training=True)
        d_tilde, l_tilde = dis(x_tilde, training=True)

        ### Compute losses ###
        ll_loss = reconstruction_loss(l_tilde, l_real)
        kl_loss = kl_divergence(mu, logvar)

        dis_loss = d_loss(d_real, d_fake, d_tilde)
        dec_loss = g_loss(d_fake, d_tilde) + 0.2*(ll_loss/(8*8*256))
        enc_loss = kl_loss + (ll_loss/(8*8*256))

        ### Compute gradients ###
        g_enc = enc_tape.gradient(enc_loss, enc.trainable_variables)
        g_dec = dec_tape.gradient(dec_loss, dec.trainable_variables)
        g_dis = dis_tape.gradient(dis_loss, dis.trainable_variables)

        ### Apply gradients ###
        enc_opt.apply_gradients(zip(g_enc, enc.trainable_variables))
        dec_opt.apply_gradients(zip(g_dec, dec.trainable_variables))
        dis_opt.apply_gradients(zip(g_dis, dis.trainable_variables))
    
    return enc_loss, dec_loss, dis_loss

def train(dataset, steps_per_epoch=10, epochs=100):
    sample_vector = K.random_normal(mean=0, stddev=1.0, shape=(1, latent_dim))
    images = []
    for i in range(epochs):
        print('[*] EPOCH #%d' % (i+1))
        for j in range(steps_per_epoch):
            x_train, y_train = next(iter(dataset))
            enc_loss, dec_loss, dis_loss = _train_step(x_train)

            print('[*] Batch #[%d/%d], enc loss = %.5f - dec loss = %.5f - dis loss = %.5f' % (j+1, steps_per_epoch, enc_loss, dec_loss, dis_loss))

        ### Predict one image to see the result ###
        sample_image = dec.predict(sample_vector)[0]
        images.append(sample_image)
        _generate_gif(images)

        ### Repeat the dataset ###
        dataset = dataset.repeat()
        if(i % 15 == 0):
            print('[INFO] Creating checkpoints ... ')
            enc.save_weights('checkpoints/enc.weights.hdf5')
            dec.save_weights('checkpoints/dec.weights.hdf5')
            dis.save_weights('checkpoints/dis.weights.hdf5')

        print('========================================================================================')

### Loading data ###
if(not os.path.exists(record_file)):
    ### Write data if not exists ###
    print('[INFO] Parsing data to tfrecord')
    write_to_tfrecord(data_dir, record_file=record_file)

print('[INFO] Loading data from tfrecord')
epochs = 1000
batch_size = 64

print('[INFO] Starting training ... ')
dataset_len, dataset = read_from_tfrecord(record_file, batch_size)
steps_per_epoch = dataset_len // batch_size
train(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch)
