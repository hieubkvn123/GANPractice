import os
import cv2
import numpy as np

import face_recognition as fr

def detect_face(img):
    ### Returns the face crop ###
    face_locations = fr.face_locations(img) # only take first face
    
    if(len(face_locations) > 0):
        top, right, bottom, left = face_locations[0]
        face = img[top:bottom, left:right]

        return face

    return None

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
            img = detect_face(img)

            if(img is None):
                continue

            #print('[*] Face detected in image %s ... ' % abs_path)
            img = cv2.resize(img, (dimensions[0], dimensions[1]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.reshape(dimensions[0], dimensions[1], dimensions[2])

            images.append(img)
            num_images += 1
    
    images = np.array(images)
    return images
