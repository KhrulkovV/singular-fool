import numpy as np
import tensorflow as tf
import os
from scipy.misc import imresize
from PIL import Image


def load_images(input_dir):
    images = []

    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):

        image = np.array(Image.open(filepath).convert('RGB')).astype(np.float)
        images.append(imresize(image, (224, 224)))

    return np.array(images)
