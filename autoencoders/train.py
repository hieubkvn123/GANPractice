import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from sparse_autoencoder import SparseAutoencoder

### Prepare the argument parser ###
parser = ArgumentParser()
parser.add_argument('--batch_size', required=False, default=256, help='Number of train images per batch')
parser.add_argument('--epochs', required=False, default=1000, help='Number of training iterations')
parser.add_argument('--lambda', required=False, default=0.001, help='Coefficient of kernel regularizer')
parser.add_argument('--sparsity', required=False, default=0.01, help='Sparsity parameter for KLD regularizer')
parser.add_argument('--beta', requied=False, default=3, help='Coefficient of activity regularizer')
parser.add_argument('--encoding_dim', required=False, default=200, help='The dimension of encoded features')
args = vars(parser.parse_args())

BATCH_SIZE=args['batch_size']
EPOCHS=args['epochs']
LAMBDA=args['lambda']
SPARSITY=args['sparsity']
BETA=args['beta']
ENCODING_DIM=args['encoding_dim']

### Prepare the dataset ###
(train_images, train_labels), (_,_)  = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
train_images = (train_images - 127.5)/127.5 ### Normalize image to [-1,1]

sparse_ae = SparseAutoencoder(lambda=LAMBDA,
                              sparsity=SPARSITY,
                              beta=BETA,
                              encoding_dim=ENCODING_DIM)
autoencoder, encoder, decoder = sparse_autoencoder.get_model()

autoencoder.fit(x=train_images, y=train_images, batch_size=BATCH_SIZE, epochs=EPOCHS)

