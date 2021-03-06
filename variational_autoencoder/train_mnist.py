import os
import numpy as np
import tensorflow as tf

from model import VAE
from argparse import ArgumentParser
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

parser = ArgumentParser()
parser.add_argument('--num_train', required=False, default=60000, 
        help='Number of training data to fit into the model')
parser.add_argument('--epochs', required=False, default=1000,
        help='Number of training iterations')
parser.add_argument('--checkpoint', required=False, default='checkpoints/model.weights.hdf5',
        help='Path the checkpoint file')
parser.add_argument('--batch_size', required=False, default=64,
        help='Number of training data to process at once')
parser.add_argument('--decoder_checkpoint', required=False, default='checkpoints/decoder.weights.hdf5',
        help='Path to the decoder weights file')
args = vars(parser.parse_args())

EPOCHS = int(args['epochs'])
BATCH_SIZE = int(args['batch_size'])
NUM_TRAIN = int(args['num_train'])
MODEL_CHECKPOINT = args['checkpoint']
DECODER_CHECKPOINT = args['decoder_checkpoint']

model, decoder = VAE().build()

if(os.path.exists(MODEL_CHECKPOINT)):
    print('[*] Loading pretrained model ... ')
    model.load_weights(MODEL_CHECKPOINT)

### Load dataset ###
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
X_train = (X_train.reshape(-1, 28,28,1) / 255.0).astype('float32')
X_test = (X_test.reshape(-1, 28, 28, 1) / 255.0).astype('float32')

print('[*] Number of training images : {}'.format(X_train.shape[0]))
print('[*] Number of testing images : {}'.format(X_test.shape[0]))

callbacks = [
    ModelCheckpoint(MODEL_CHECKPOINT, save_best_only=True, verbose=1),
    EarlyStopping(patience=15, verbose=1),
    CSVLogger('training.log.csv')
]

Y = {
    'decoded_output' : X_train,
    'encoded_output' : np.ones((X_train.shape[0], 512))
}

Y_ = {
    'decoded_output' : X_test,
    'encoded_output' : np.ones((X_test.shape[0], 512))
}

print(model.summary())
try:
    model.fit(X_train, Y,
        callbacks=callbacks,
        validation_data=(X_test, Y_),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE)
except KeyboardInterrupt:
    print('[*] Saving decoder ... ')
    model.get_layer('decoder').save_weights(DECODER_CHECKPOINT)
