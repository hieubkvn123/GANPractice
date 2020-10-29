import cv2
import pickle
import numpy as np 

from denoise_autoencoder import DenoiseAutoencoder

MODEL_CHECKPOINT = 'checkpoints/autoencoder.weights.hdf5'
TEST_PICKLE = 'data/X_test.pickle'
net = DenoiseAutoencoder()
autoencoder = net.build()

print('[*] Loading model ... ')
autoencoder.load_weights(MODEL_CHECKPOINT)
print(autoencoder.summary())

X_test = pickle.load(open(TEST_PICKLE, 'rb'))
np.random.shuffle(X_test)

test_image = X_test[0]
original = test_image
H, W, C = test_image.shape

test_image = cv2.resize(test_image, (128,128))
test_image = np.array([test_image])
test_image = test_image / 255.0
test_image = test_image.astype('float32')

reconstructed = autoencoder.predict(test_image)[0]
reconstructed = reconstructed.astype(np.float32)
#reconstructed = cv2.resize(reconstructed, (W, H))
reconstructed = reconstructed.astype('float32')

cv2.imshow('Original', original)
cv2.imshow('Denoised', reconstructed)

key = cv2.waitKey(0)
if(key == ord("q")):
    cv2.destroyAllWindows()
