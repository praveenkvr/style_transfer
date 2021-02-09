import tensorflow as tf
import numpy as np

from utils import gram_matrix


class InstanceNorm(tf.keras.layers.Layer):

    def __init__(self, epsilon=1e-3):

        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.beta = tf.Variable(tf.zeros([input_shape[3]]))
        self.gamma = tf.Variable(tf.ones([input_shape[3]]))

    def call(self, inputs):

        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        x = tf.divide(tf.subtract(inputs, mean),
                      tf.sqrt(tf.add(variance, self.epsilon)))

        return self.gamma * x + self.beta


def create_vgg_model(layer_names):
    """
    Create VGG model with imagenet weights
    """
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.models.Model(inputs=vgg.inputs, outputs=outputs)
    return model


def conv2d_block(input_tensor, filter, kernel_size, strides, res=False, activation=True):

    x = input_tensor
    pad = kernel_size // 2
    paddings = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
    x = tf.pad(x, paddings, mode='REFLECT')
    x = tf.keras.layers.Conv2D(filter, kernel_size=kernel_size,
                               strides=strides, padding="valid", use_bias=False)(x)
    x = InstanceNorm()(x)
    if activation:
        x = tf.keras.layers.Activation('relu')(x)
    residual_block = x

    if res:
        x = tf.keras.layers.Conv2D(
            filter, kernel_size=kernel_size, strides=strides, padding="same", use_bias=False)(x)
        # x = InstanceNorm()(x)
        x = tf.keras.layers.add([residual_block, x])

    return x


def deconv_block(input_tensor, filter):

    x = input_tensor

    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    x = tf.keras.layers.Conv2D(filter, kernel_size=(
        3, 3), padding='same', strides=(1, 1))(x)
    x = InstanceNorm()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def custom_actication(x):
    return (tf.nn.tanh(x) * 150 + 255.0/2)


def custom_unet_model(input_shape=(None, None, 3)):

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = conv2d_block(inputs, 32, 11, 1)
    x = conv2d_block(x, 64, 3, 2)
    x = conv2d_block(x, 128, 3, 2)

    x = conv2d_block(x, 128, 3, 1, res=True)
    x = conv2d_block(x, 128, 3, 1, res=True)
    x = conv2d_block(x, 128, 3, 1, res=True)
    x = conv2d_block(x, 128, 3, 1, res=True)
    x = conv2d_block(x, 128, 3, 1, res=True)

    x = deconv_block(x, 128)
    x = deconv_block(x, 64)

    x = conv2d_block(x, 3, 11, 1, activation=False)
    x = (tf.nn.tanh(x) * 150 + 255./2)
    model = tf.keras.models.Model(inputs, x)
    return model


def preprocess_input(x):

    mean = np.array([123.68, 116.779, 103.939])
    return (x - mean)


class ContentStyleModel(tf.keras.layers.Layer):

    def __init__(self, content_layers, style_layers):
        super(ContentStyleModel, self).__init__()

        self.vgg = create_vgg_model(style_layers + content_layers)
        self.content_layers = content_layers
        self.style_layers = style_layers

        self.content_layers_len = len(content_layers)
        self.style_layers_len = len(self.style_layers)

        self.vgg.trainable = False

    def call(self, inputs):
        preprocessed = preprocess_input(inputs)
        outputs = self.vgg(preprocessed)

        style_outputs = [gram_matrix(features)
                         for features in outputs[:self.style_layers_len]]
        content_outputs = outputs[self.style_layers_len:]

        return style_outputs, content_outputs
