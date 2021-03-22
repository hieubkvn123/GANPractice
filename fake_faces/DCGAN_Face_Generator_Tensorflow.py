#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[10]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Tensorflow
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers

# Some constants
lr = 0.0002
batch_size = 64
epochs = 100
img_shape = (112, 112, 3)
latent_dim = 100


# # Building the models

# ## 1. Generator

# In[44]:


def make_generator():
    l2 = regularizers.l2(l2=1e-4)
    inputs = Input(shape=(latent_dim,))
    
    x = Dense(7*7*256, use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Reshape(target_shape=(7,7,256))(x)
    
    # Size = 128 x 14 x 14 
    x = Conv2DTranspose(128, kernel_size=(5,5), strides=(2,2), padding='same',
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    # Size = 64 x 28 x 28
    x = Conv2DTranspose(64, kernel_size=(5,5), strides=(2,2), padding='same',
                       use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    # Size = 32 x 56 x 56
    x = Conv2DTranspose(32, kernel_size=(5,5), strides=(2,2), padding='same',
                       use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    # Size = 16 x 112 x 112
    x = Conv2DTranspose(16, kernel_size=(5,5), strides=(2,2), padding='same',
                       use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    # Use tanh so that it is [-1, 1]
    x = Conv2D(img_shape[-1], kernel_size=(5,5), padding='same', use_bias=False, activation='tanh')(x)
    
    model = Model(inputs=inputs, outputs=x, name='Generator')
    return model

G = make_generator()
G.summary()


# ## 2. Discriminator

# In[45]:


def make_discriminator():
    inputs = Input(shape=img_shape)
    
    x = Conv2D(16, kernel_size=(5,5), strides=(2,2), padding='same')(inputs)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(32, kernel_size=(5,5), strides=(2,2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(64, kernel_size=(5,5), strides=(2,2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(128, kernel_size=(5,5), strides=(2,2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    
    x = Flatten()(x)
    x = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=x, name='Discriminator')
    return model

D = make_discriminator()
D.summary()


# # Definition of loss functions

# ## 1. Generator Loss

# In[46]:


bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
'''
    Evaluates how well the Generator can deceive the Discriminator
    Calculated by taking the binary cross entropy of D(G(z)) w.r.t ones
'''
def generator_loss(D_fake):
    return bce(tf.ones_like(D_fake), D_fake)


# ## 2. Discriminator Loss

# In[40]:


'''
    Evaluates how well the discriminator is able to separate true and fake instances
    Calculated by taking the sume of bce(D(G(z)), 0) and bce(D(x), 1)
'''
def discriminator_loss(D_true, D_fake):
    fake_loss = K.binary_crossentropy(tf.zeros_like(D_fake), D_fake)
    true_loss = K.binary_crossentropy(tf.ones_like(D_true), D_true)
    
    return fake_loss + true_loss


# # Preparing the dataset

# In[41]:


def preprocess(img):
    # normalize
    img = (img - 127.5) / 127.5
    img = img.astype(np.float32)
    
    return img
    
generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess)
dataset = generator.flow_from_directory("/home/minhhieu/Desktop/Hieu/datasets/img_align_celeba",
                                       target_size=(img_shape[0], img_shape[1]),
                                       batch_size=batch_size)
dataset_size = len(dataset)
steps_per_epoch = dataset_size // batch_size

# # Start the training process

# In[49]:


# Define the optimizers
g_opt = optimizers.Adam(learning_rate=lr, amsgrad=True)
d_opt = optimizers.Adam(learning_rate=lr, amsgrad=True)

# Define training step per batch
def train_step(images):
    noise = tf.random.normal([batch_size, latent_dim])
    
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_img = G(noise, training=True)
        
        real_output = D(fake_img, training=True)
        fake_output = D(images, training=True)
        real_output = K.clip(real_output, K.epsilon(), 1 - K.epsilon())
        fake_output = K.clip(fake_output, K.epsilon(), 1 - K.epsilon())
        
        print(fake_output, real_output)
        g_loss = generator_loss(fake_output)
        d_loss = discriminator_loss(real_output, fake_output)
        
        g_loss = K.mean(g_loss)
        d_loss = K.mean(d_loss)
        
    print(g_loss)
    d_gradient = d_tape.gradient(d_loss, D.trainable_variables)
    g_gradient = g_tape.gradient(g_loss, G.trainable_variables)
    print(g_gradient)
    
    d_opt.apply_gradients(zip(d_gradient, D.trainable_variables))
    g_opt.apply_gradients(zip(g_gradient, G.trainable_variables))
    
    return g_loss, d_loss


# In[50]:


# Define the training loop
for epoch in range(epochs):
    for batchX, batchY in dataset:
        g_loss, d_loss = train_step(batchX)
        print(g_loss, d_loss)


# In[9]:


get_ipython().system('pip3 list | grep tensorflow')


# In[ ]:




