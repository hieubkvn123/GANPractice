import os
import cv2
import numpy as np

### Load data from folder ###
### resize images to a specific dimensions ###
def data_from_dir(data_dir, dimensions=(128,128,1), batch_size=32, max_num_img=None):
    images = list()

    if(not max_num_img):
        max_num_img = 1000

    num_images = 0
    for (dir, dirs, files) in os.walk(data_dir):
        for file in files:
            if(num_images >= max_num_img):
                break

            abs_path = dir + "/" + file
            img = cv2.imread(abs_path)
            img = cv2.resize(img, (dimensions[0], dimensions[1]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.reshape(dimensions[0], dimensions[1], dimensions[2])

            images.append(img)
            num_images += 1
    
    images = np.array(images)
    return images
