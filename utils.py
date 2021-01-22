
import tensorflow as tf
import numpy as np

IMG_WIDTH = 224
IMG_HEIGHT = 224

def gram_matrix(tensor):

    batch_size, width, height, filters = tensor.shape
    features = tf.reshape(tensor, (batch_size, width * height, filters))
    features_t = tf.transpose(features, perm=[0, 2, 1])
    gram = tf.matmul(features, features_t)

    return gram


def load_image(path, add_dim=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    if add_dim:
        img = img[np.newaxis, :]

    return img
