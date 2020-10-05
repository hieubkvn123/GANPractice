import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from models import make_generator, make_discriminator

g = make_generator()
d = make_discriminator()

noise = tf.random.normal([1, 100])
generated_image = g(noise)
decision = d(generated_image)

print(decision)
plt.imshow(generated_image[0])
plt.show()
