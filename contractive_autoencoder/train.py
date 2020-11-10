import os
import pickle
import numpy as np
import tensorflow as tf

from argparse import ArgumentParser
from data_loader import data_from_dir
from model import ContractiveAutoencoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

parser = ArgumentParser()
parser.add_argument('-d', '--data_dir', required=True, 
        help='Path to the image data folder')
parser.add_argument('-n', '--num_train', required=False, default=10000,
        help='Number of training images (test images inclusive)')
parser.add_argument('-e', '--epochs', required=False, default=1000,
        help='Number of training iterations')
parser.add_argument('-c', '--checkpoint', required=False, default='./checkpoints',
        help='The location of the checkpoint directory')
parser.add_argument('-b', '--batch_size', required=False, default=16,
        help='Number of images to process per batch per epoch')
args = vars(parser.parse_args())

DATA_DIR = args['data_dir']
PICKLE_DIR = 'data/'
CKPT_DIR = args['checkpoint']
EPOCHS = int(args['epochs'])
BATCH_SIZE = int(args['batch_size'])
NUM_TRAIN = int(args['num_train'])

autoencoder = ContractiveAutoencoder()
model = autoencoder.build() ### Normal strategy with 1 gpu ###

### IMPORTANT : How to train with multi-gpu ###
num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
if(num_gpus > 1):
    print('[*] Training in Multi-GPU mode ... ')
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = autoencoder.build()

print(model.summary())


if(len(os.listdir(PICKLE_DIR)) == 0):
    X_train, X_test, Y_train, Y_test = data_from_dir(DATA_DIR, NUM_TRAIN)
    
    print('[*] Serializing data ... ')
    pickle.dump(X_train, open(os.path.join(PICKLE_DIR, 'X_train.pickle'), 'wb'))
    pickle.dump(X_test, open(os.path.join(PICKLE_DIR, 'X_test.pickle'), 'wb'))
    pickle.dump(Y_train, open(os.path.join(PICKLE_DIR, 'Y_train.pickle'), 'wb'))
    pickle.dump(Y_test, open(os.path.join(PICKLE_DIR, 'Y_test.pickle'), 'wb'))
else:
    X_train = pickle.load(open(os.path.join(PICKLE_DIR, 'X_train.pickle'), 'rb'))
    X_test = pickle.load(open(os.path.join(PICKLE_DIR, 'X_test.pickle'), 'rb'))
    Y_train = pickle.load(open(os.path.join(PICKLE_DIR, 'Y_train.pickle'), 'rb'))
    Y_test = pickle.load(open(os.path.join(PICKLE_DIR, 'Y_test.pickle'), 'rb'))

print("[*] Number of training images : {}, full shape : {}".format(X_train.shape[0], str(X_train.shape)))
print("[*] Number of testing images  : {}, full shape : {}".format(X_test.shape[0], str(X_test.shape)))
print("[*] Training labels shape : {}".format(Y_train.shape))

### Normalize images to [0,1] range ###
X_train = (X_train[:NUM_TRAIN]/255.0).astype('float32')
X_test  = (X_test[:NUM_TRAIN]/255.0).astype('float32')
Y_train = (Y_train[:NUM_TRAIN]/255.0).astype('float32')
Y_test  = (Y_test[:NUM_TRAIN]/255.0).astype('float32')

callbacks = [
    ModelCheckpoint(os.path.join(CKPT_DIR, 'autoencoder.weights.hdf5'), verbose=1, save_best_only=True),
    EarlyStopping(patience=15, verbose=1),
    CSVLogger('training.log.csv')
]

if(len(os.listdir(CKPT_DIR)) > 0):
    print('[*] Loading checkpoint ...')
    model.load_weights(os.path.join(CKPT_DIR, 'autoencoder.weights.hdf5'))

Y = {
    'decoded' : Y_train,
    'encoded' : np.ones(Y_train.shape[0], 128)
}


Y_ = {
    'decoded' : Y_test,
    'encoded' : np.ones(Y_test.shape[0], 128)
}

losses = {
    'encoded' : model.contractive_loss,
    'decoded' : tf.nn.mean_squared_error
}

model.compile(optimizer='adam', loss=losses)

model.fit(X_train, Y, 
        validation_data=(X_test, Y_),
        callbacks=callbacks,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE)

