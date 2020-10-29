import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sparse_autoencoder import SparseAutoencoder
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model

AUTOENCODER_CKPT = 'checkpoints/sparse_autoencoder_1.weights.hdf5'
DECODER_CKPT = 'checkpoints/sparse_decoder.weights.hdf5'
ENCODER_CKPT = 'checkpoints/sparse_encoder.weights.hdf5'
NUM_IM_PER_CLASS = 20
IMG_DIR = 'reconstructed'

### output images embeddings in a scatter plot ###
### and output reconstructed images ###

net = SparseAutoencoder()
autoencoder = net.build()
autoencoder.load_weights(AUTOENCODER_CKPT)

(train_images, train_labels), (_,_)  = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
train_images = train_images.astype("float32")/255.0 ### Normalize image to [-1,1]

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
### Visualize the reconstructed inputs ###
reconstructed = autoencoder.predict(test_images)
reconstructed = reconstructed * 255.0
reconstructed = reconstructed.astype(np.uint8)

if(not os.path.exists(IMG_DIR)):
    print('[*] Creating reconstructed input image directory ...')
    os.mkdir(IMG_DIR)


### Create 2 plots, one for original, one for recontructed ###
fig1, axes1 = plt.subplots(4,3, figsize=(6,6))
fig2, axes2 = plt.subplots(4,3, figsize=(6,6))

for i, label in enumerate(np.unique(test_labels)):
    if(not os.path.exists('{}/{}'.format(IMG_DIR, label))):
        print('[*] Creating reconstructed dir for label {}'.format(label))
        os.mkdir('{}/{}'.format(IMG_DIR, label))

    original = test_images[test_labels == label]
    cluster = reconstructed[test_labels == label]

    random_index = np.random.randint(0, cluster.shape[0]) 
    axes1[i//3][i%3].imshow(cluster[random_index])
    axes2[i//3][i%3].imshow(original[random_index])
    
    for i, image in enumerate(cluster):
        cv2.imwrite('{}/{}/{}.png'.format(IMG_DIR, label, i), image)

fig1.suptitle('Reconstructed inputs')
fig2.suptitle('Original inputs')
plt.show()
