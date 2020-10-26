import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sparse_autoencoder import SparseAutoencoder
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model

AUTOENCODER_CKPT = 'checkpoints/sparse_autoencoder.weights.hdf5'
DECODER_CKPT = 'checkpoints/sparse_decoder.weights.hdf5'
ENCODER_CKPT = 'checkpoints/sparse_encoder.weights.hdf5'
NUM_IM_PER_CLASS = 20
IMG_DIR = 'reconstructed'

### output images embeddings in a scatter plot ###
### and output reconstructed images ###

net = SparseAutoencoder()
autoencoder, encoder, decoder = net.get_model()
autoencoder.load_weights(AUTOENCODER_CKPT)
encoder = Model(autoencoder.inputs, autoencoder.layers[-3].output)
# encoder.load_weights(ENCODER_CKPT)

(train_images, train_labels), (_,_)  = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
train_images = (train_images - 127.5)/127.5 ### Normalize image to [-1,1]

test_images = np.array([])
test_labels = np.array([])
### Randomly choosing images from all classes ###
for label in np.unique(train_labels):
    images = train_images[train_labels == label]
    
    # shuffle the chosen images and extract a certain number of them
    np.random.shuffle(images)
    images = images[:NUM_IM_PER_CLASS]
    labels = np.full((NUM_IM_PER_CLASS,), label)

    if(len(test_images) == 0 and len(test_labels) == 0):
        test_images = images
        test_labels = labels
    else:
        test_images = np.concatenate((test_images, images))
        test_labels = np.concatenate((test_labels, labels))

### PCA to reduce encodings for visualisation ###
def pca(x, n_components=3):
    ### First, standardize the dataset column wise ###
    x = (x - np.mean(x, axis=0))/np.std(x, axis=0)
    ### get the covariance matrix ###
    covar = np.cov(x.transpose())
    eig_val, eig_vec = np.linalg.eig(covar)

    ### sort the eigen values in descending order ###
    eig_val_indices = eig_val.argsort()[::-1]

    ### sort the eigen vectors in eigen values order ###
    eig_vec = eig_vec[eig_val_indices]

    ### get the top eigen vectors ###
    top_eig_vec = eig_vec[:, :n_components]

    final = np.dot(x, top_eig_vec)

    return final

### Visualize the encodings in a 3D scatter ###
encodings = encoder.predict(test_images)
pca_enc = pca(encodings) 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for label in np.unique(test_labels):
    cluster = pca_enc[test_labels == label]
    ax.scatter3D(cluster[:,0], cluster[:,1], cluster[:,2], label=label)

fig.savefig("encodings.png")

### Visualize the reconstructed inputs ###
reconstructed = autoencoder.predict(test_images)
reconstructed = reconstructed * 127.5 + 127.5
reconstructed = reconstructed.astype(np.uint8)

if(not os.path.exists(IMG_DIR)):
    print('[*] Creating reconstructed input image directory ...')
    os.mkdir(IMG_DIR)

for label in np.unique(test_labels):
    if(not os.path.exists('{}/{}'.format(IMG_DIR, label))):
        print('[*] Creating reconstructed dir for label {}'.format(label))
        os.mkdir('{}/{}'.format(IMG_DIR, label))

    cluster = reconstructed[test_labels == label]
    
    for i, image in enumerate(cluster):
        cv2.imwrite('{}/{}/{}.png'.format(IMG_DIR, label, i), image)
