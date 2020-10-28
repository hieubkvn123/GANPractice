import os
import numpy as np
import tensorflow as tf

from argparse import ArgumentParser
from data_loader import data_from_dir
from denoise_autoencoder import DenoiseAutoencoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

parser = ArgumentParser()
parser.add_argument('-d', '--data_dir', required=True, 
        help='Path to the image data folder')
parser.add_argument('-n', '--num_train', required=False, default=10000,
        help='Number of training images (test images inclusive)')
args = vars(parser.parse_args())

DATA_DIR = args['data_dir']
NUM_TRAIN = int(args['num_train'])

net = DenoiseAutoencoder()
autoencoder, encoder, decoder = net.build()

print(autoencoder.summary())
X_train, X_test, Y_train, Y_test = data_from_dir(DATA_DIR, NUM_TRAIN)

print("[*] Number of training images : {}".format(X_train.shape[0]))
print("[*] Number of testing images  : {}".format(X_test.shape[0]))


