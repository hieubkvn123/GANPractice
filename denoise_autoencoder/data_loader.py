import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

def awgn(x, sigma=5):
    H, W, C = x.shape
    mean = 0
    
    noise = np.random.normal(mean, sigma, (H, W, C))
    noisy = x + noise
    noisy = noisy.astype(np.uint8)

    return noisy

def speckle(x, sigma=0.1):
    ### Similar to awgn but y = x + x * z ###
    H, W, C = x.shape
    mean = 0
    
    noise = np.random.normal(mean, sigma, (H, W, C))
    noisy = x + np.multiply(x ,noise)
    noisy = noisy.astype('uint8')

    return noisy

def data_from_dir(data_dir, num_train, preprocessing=[awgn, speckle], 
        test_size=0.3333, input_shape=(128,128,3)):
    counter = 0
    train_images = []
    train_labels = []
    for (dir_, dirs, files) in os.walk(data_dir):
        for file_ in files:
            counter += 1
            abs_path = os.path.join(dir_, file_)
            image = cv2.imread(abs_path)
            label = image

            for prep in  preprocessing:
                noisy = prep(image)
                train_images.append(noisy)
                train_labels.append(label)

            print('[*] Processing image : {}'.format(abs_path))

            if(counter >= num_train):
                break

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    X_train, X_test, Y_train, Y_test = train_test_split(train_images, train_labels, test_size=test_size, shuffle=True)

    return X_train, X_test, Y_train, Y_test
