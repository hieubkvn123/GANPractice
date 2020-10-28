import cv2
import pickle
import numpy as np 

from denoise_autoencoder import DenoiseAutoencoder

MODEL_CHECKPOINT = 'checkpoints/autoencoder.weights.hdf5'
TEST_PICKLE = 'data/X_test.pickle'
net = DenoiseAutoencoder()
autoencoder, encoder, decoder = net.build()

print('[*] Loading model ... ')
autoencoder.load_weights(MODEL_CHECKPOINT)
print(autoencoder.summary())

X_test = pickle.load(open(TEST_PICKLE, 'rb'))
np.random.shuffle(X_test)
test_image = X_test[0]

reconstructed = model.predict(test_image)
reconstructed = reconstructed * 255.0
reconstructed = reconstructed.astype(np.uint8)

cv2.imshow('Original', test_image)
cv2.imshow('Denoised', reconstructed)

key = cv2.waitKey(0)
if(key == ord("q")):
    cv2.destroyAllWindows()
