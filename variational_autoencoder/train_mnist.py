import os
import numpy as np
import tensorflow as tf

from model import VAE
from argparse import ArgumentParser
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

parser = ArgumentParser()
parser.add_argument('--num_train', required=False, default=10000, 
        help='Number of training data to fit into the model')
parser.add_argument('--epochs', required=False, default=1000,
        help='Number of training iterations')
parser.add_argument('--checkpoint', required=False, default='checkpoints/model.weights.hdf5',
        help='Path the checkpoint file')
parser.add_argument('--batch_size', required=False, default=64,
        help='Number of training data to process at once')
args = vars(parser.parse_args())

EPOCHS = int(args['epochs'])
BATCH_SIZE = int(args['batch_size'])
NUM_TRAIN = int(args['num_train'])
MODEL_CHECKPOINT = args['checkpoint']

net = VAE().build()

if(os.path.exists(MODEL_CHECKPOINT)):
    net.load_weights(MODEL_CHECKPOINT)

### Load dataset ###
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 28,28,1) / 255.
X_test = X_test.reshape(-1, 28, 28, 1) / 255.


