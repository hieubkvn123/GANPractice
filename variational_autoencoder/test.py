import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from model import VAE
from scipy.stats import norm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--decoder', required=False, default='checkpoints/decoder.weights.hdf5',
        help='Path to the decoder weights file')
args = vars(parser.parse_args())

DECODER_CHECKPOINT = args['decoder']

model, decoder = VAE().build()
if(os.path.exists(DECODER_CHECKPOINT)):
    print('[*] Loading decoder weights file from checkpoint')
    decoder.load_weights(DECODER_CHECKPOINT)


def decode(eps, apply_sigmoid=False):
    logits = decoder.predict(eps)
    if(apply_sigmoid):
        probs = tf.sigmoid(logits)
        return probs
    return logits

def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, 2))
    return decode(eps, apply_sigmoid=True)

### Generate vectors from N(1,0) ###
#sample = np.random.normal(0, 1, (225, 2))

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
      x_decoded = sample(z)
      digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
      image[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit.numpy()

  plt.figure(figsize=(10, 10))
  plt.imshow(image, cmap='Greys_r')
  plt.axis('Off')
  plt.savefig('predictions.png')

plot_latent_images(decoder, 10)
'''
predictions = decoder.predict(sample)
fig, ax = plt.subplots(15,15, figsize=(45,45))
for i in range(15*15):
    row = i // 15
    col = i %  15
    ax[row][col].imshow(predictions[i])

print(predictions)
plt.savefig('predictions.png')
'''
