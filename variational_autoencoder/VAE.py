#!/usr/bin/env python
# coding: utf-8

# In[3]:


### First, build the model ###
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, regularizers

class VAE(object):
    def __init__(self, input_shape=(28,28,1), latent_dim=2):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
    def kl_divergence(self, y_true, y_pred):
        mean, logvar = tf.split(y_pred, num_or_size_splits=2, axis=1)
        
        kl_loss = -0.5 * K.sum(1 + logvar - 
                              K.square(mean) - 
                              K.exp(logvar), axis=-1)
        
        return kl_loss
        
    def build_decoder(self):
        inputs = Input(shape=(self.latent_dim,))
        d1 = Dense(7 * 7 * 16, activation='relu')(inputs)
        
        reshape = Reshape(target_shape=(7,7,16))(d1)
        
        ### Now dimension = 14 x 14 x 16 ###
        conv1 = Conv2DTranspose(16, activation='relu', kernel_size=3, strides=2, padding='same')(reshape)
        ### Now dimension = 28 x 28 x 8 ###
        conv2 = Conv2DTranspose(8, activation='relu', kernel_size=3, strides=2, padding='same')(conv1)
        output = Conv2DTranspose(1, kernel_size=3, strides=1, padding='same')(conv2)
        
        decoder = Model(inputs, output, name='Decoder')
        return decoder
    
    def build_encoder(self):
        inputs = Input(shape=self.input_shape)
        ### Now dimension = 14 x 14 x 8 ###
        conv1 = Conv2D(8, kernel_size=3, strides=(2,2), activation='relu')(inputs)
        ### Now dimension = 7 x 7 x 16 ###
        conv2 = Conv2D(16, kernel_size=3, strides=(2,2), activation='relu')(conv1)
        flatten = Flatten()(conv2)
        output = Dense(self.latent_dim + self.latent_dim)(flatten)
        
        encoder = Model(inputs, output, name='Encoder')
        return encoder
        
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=(self.latent_dim,))
        return eps * tf.exp(logvar * 0.5) + mean
        
    def build(self):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
        inputs = Input(shape=self.input_shape)
        encoded = self.encoder(inputs)
        
        ### Reparameterize and decode ###
        ### z = eps * sigma + mu ###
        mean, logvar = tf.split(encoded, num_or_size_splits=2, axis=1)
        z = self.reparameterize(mean, logvar)
        
        decoded = self.decoder(z)
        
        ### Rename the two tensors to assign losses ###
        encoded = Lambda(lambda x : x, name='encoded_output')(encoded)
        decoded = Lambda(lambda x : x, name='decoded_output')(decoded)
        model = Model(inputs, [decoded, encoded])
        
        return model
    
model = VAE().build()
print(model.summary())


# In[4]:


### Then prepare the dataset ###
(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()

### Normalize the dataset to [-1, 1] range ###
def normalize_data(x):
    x = (x - 127.5) / 127.5
    return x
def denormalize_data(x):
    x = x * 127.5 + 127.5
    x = x.astype(np.uint8)
    return x

X_train = normalize_data(X_train)
X_test = normalize_data(X_test)


# In[ ]:


### Prepare training phases ###
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

INPUT_SHAPE = (28,28,1)
LATENT_DIM = 2
BATCH_SIZE=100
EPOCHS=1500
MODEL_CHECKPOINT = 'model_1.weights.hdf5'

vae = VAE(input_shape=INPUT_SHAPE, latent_dim=LATENT_DIM)
model = vae.build()
if(os.path.exists(MODEL_CHECKPOINT)):
    print('[*] Transfer learning from existing checkpoint ...')
    model.load_weights(MODEL_CHECKPOINT)

Y_train = {
    'decoded_output' : X_train,
    'encoded_output' : np.zeros((X_train.shape[0], LATENT_DIM * 2))
}

Y_test = {
    'decoded_output' : X_test,
    'encoded_output' : np.zeros((X_test.shape[0], LATENT_DIM * 2))
}

adam = Adam(lr=1e-2, amsgrad=True)
losses = {
    'decoded_output' : tf.keras.losses.mean_squared_error,
    'encoded_output' : vae.kl_divergence
}

def lr_decay(epochs, lr):
    decay_rate = 1 / EPOCHS
    init_lr = 1e-2 
    
    return init_lr * (1.0/(1.0 + decay_rate * epochs))

callbacks = [
    ModelCheckpoint(MODEL_CHECKPOINT, verbose=1, save_best_only=True),
    EarlyStopping(patience=15, verbose=1),
    CSVLogger('training.log.csv'),
    LearningRateScheduler(lr_decay, verbose=1)
]

model.compile(optimizer=adam, loss=losses)
model.fit(X_train, Y_train,
         callbacks=callbacks,
         validation_data=(X_test, Y_test),
         batch_size=BATCH_SIZE,
         epochs=EPOCHS)


# In[34]:


### Now test if the model is good ###
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

def decode(model, z, apply_sigmoid=False):
    logits = model(z)
    if apply_sigmoid:
        probs = tf.sigmoid(logits)
        return probs
    return logits

def sample(model, eps=None):
    if eps is None:
        eps = tf.random.normal(shape=(100, self.latent_dim))
    return decode(model, eps, apply_sigmoid=True)

def plot_latent_images(model, n, digit_size=28):
    """Plots n x n digit images decoded from the latent space."""

    norm = tfp.distributions.Normal(0, 1)
    grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
    grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
    image_width = digit_size*n
    image_height = image_width
    image = np.zeros((image_height, image_width))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
          z = np.array([[xi, yi]])
          x_decoded = sample(model,z)
          digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
          image[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit.numpy()
    
    image = denormalize_data(image)
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='Greys_r')
    plt.axis('Off')
    plt.savefig('predictions.png')
    # plt.show() 

vae = VAE().build()
if(os.path.exists(MODEL_CHECKPOINT)):
    print('[*] Found existing model checkpoint ... ')
    vae.load_weights(MODEL_CHECKPOINT)
decoder = vae.get_layer('Decoder')
plot_latent_images(decoder, 20)


# In[ ]:




