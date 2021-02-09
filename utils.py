
import tensorflow as tf
import numpy as np
import PIL.Image

IMG_WIDTH = 256
IMG_HEIGHT = 256


def gram_matrix(tensor):

    # batch_size, width, height, filters = tensor.shape
    # features = tf.reshape(tensor, (batch_size, width * height, filters))
    # features_t = tf.transpose(features, perm=[0, 2, 1])
    # gram = tf.matmul(features, features_t)
    gram = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    input_shape = tf.shape(tensor)
    num_locations = tf.cast(
        input_shape[1] * input_shape[2] * input_shape[3], tf.float32)
    return gram/num_locations


def load_image(path, add_dim=False, resize=True):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    if resize:
        img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    if add_dim:
        img = tf.expand_dims(img, axis=0)

    return img


def style_loss(style_output, style_target):
    style_loss = tf.add_n([
        tf.reduce_mean(tf.square(style_output[i] - style_target[i])) for i in range(len(style_target))
    ])
    return style_loss


def clip_image(img):
    return tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=255.0)


def tensor_to_imgarr(tensor):
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) == 4:
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)
