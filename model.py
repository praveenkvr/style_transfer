import tensorflow as tf

from utils import gram_matrix


class InstanceNorm(tf.keras.layers.Layer):

    def __init__(self, epsilon=1e-3):

        super(InstanceNorm, self).__init__()
        self.epsilon=epsilon
        
    def build(self, input_shape):

        self.beta = tf.Variable(tf.zeros([input_shape[3]]))
        self.gaama = tf.Variable(tf.zeros([input_shape[3]]))

    def call(self, inputs):

        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        x = tf.divide(tf.subtract(inputs, mean), tf.add(variance, self.epsilon))
        return x


def create_vgg_model(layer_names):
    """
    Create VGG model with imagenet weights
    """
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.models.Model(inputs=vgg.inputs, outputs=outputs)
    return model


def conv2d_block(input_tensor, filters):

    x = input_tensor
    filter1, filter2, filter3 = filters

    x = tf.keras.layers.Conv2D(filter1, kernel_size=(3, 3), strides=(1,1), padding="same", activation='relu')(x)
    x = InstanceNorm()(x)

    x = tf.keras.layers.Conv2D(filter2, kernel_size=(3, 3), strides=(1,1), padding="same", activation='relu')(x)
    x = InstanceNorm()(x)

    x = tf.keras.layers.Conv2D(filter3, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(x)
    return x


def bottleneck_block(inputs, filter):

    x = tf.keras.layers.Conv2D(filter, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="bottleneck")(inputs)
    return x

def deconv_block(input_tensor, filters):
    
    x = input_tensor

    filter1, filter2 = filters

    x = tf.keras.layers.Conv2D(filter1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(filter2, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    return x
    

def custom_unet_model(input_shape):
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = conv2d_block(inputs, [64, 64, 128])
    x = conv2d_block(x, [128, 128, 256])
    x = conv2d_block(x, [128, 128, 256])
    x = conv2d_block(x, [256, 256, 512])

    bottleneck = bottleneck_block(x, 512)

    x = deconv_block(bottleneck, [512, 256])
    x = deconv_block(x, [256, 256])
    x = deconv_block(x, [256, 128])
    x = deconv_block(x, [128, 64])

    x = tf.keras.layers.Conv2D(3, kernel_size=(2, 2), padding="same")(x)
    
    model = tf.keras.models.Model(inputs, x)
    return model


class ContentStyleModel(tf.keras.layers.Layer):

    def __init__(self, content_layers, style_layers):
        super(ContentStyleModel, self).__init__()

        self.vgg = create_vgg_model(style_layers + style_layers)
        self.content_layers = content_layers
        self.style_layers = style_layers

        self.content_layers_len = len(content_layers)
        self.style_layers_len = len(self.style_layers)
        
        for layer in self.vgg.layers:
            layer.trainable = False

    def call(self, inputs):        
        preprocessed = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed)

        style_outputs = [gram_matrix(features) for features in outputs[0:self.style_layers_len]]
        content_outputs = outputs[self.style_layers_len:]

        return style_outputs, content_outputs
        
