import numpy as np
import tensorflow as tf

from argparse import ArgumentParser
from losses import generator_loss, discriminator_loss
from tensorflow.keras.optimizers import Adam, RMSProp

parser = ArgumentParser()
parser.add_argument('--epochs', required=True, default=1000, help='Number of training iterations')
parser.add_argument('--batch_size', required=True, default=32, help='Number of images per batch')
