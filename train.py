import os
import tensorflow as tf
import numpy as np
from utils import load_image
from tqdm import tqdm
from model import ContentStyleModel, custom_unet_model
from matplotlib import pylab as plt
from utils import tensor_to_imgarr, load_image, clip_image


WEIGHTS_PATH = os.path.join(os.getcwd(), 'weights')
PREDICTED_FILE_NAME = 'new.jpg'
TEST_IMAGE_IMAGE_PATH = os.path.join(os.getcwd(), 'dataset', 'test.jpg')

DATASET_PATH = os.path.join(os.getcwd(), 'dataset', 'val2017')
STYLE_IMAGE_PATH = os.path.join(os.getcwd(), 'dataset', 'style2.jpg')
WEIGHTS_PATH = os.path.join(os.getcwd(), 'weights')
TRAIN_INPUT_SHAPE = (None, None, 3)
BUFFER_SIZE = 8
BATCH_SIZE = 4
EPOCHS = 100
CONTENT_WEIGHT = 5
STYLE_WEIGHT = 90
TV_WEIGHT = 25


def style_loss(style_output, style_target):
    style_loss = tf.add_n([
        tf.reduce_mean(
            tf.square(style_output[i] - style_target[i])
        ) for i in range(len(style_target))
    ])

    return style_loss / len(style_output)


def content_loss(content_output, content_target):

    content_loss = tf.add_n([
        tf.reduce_mean(
            tf.square(content_output[i] - content_target[i])
        ) for i in range(len(content_target))
    ])
    return content_loss / len(content_output)


def total_variation_loss(img):
    x_var = img[:, :, 1:, :] - img[:, :, :-1, :]
    y_var = img[:, 1:, :, :] - img[:, :-1, :, :]
    return tf.reduce_mean(tf.square(x_var)) + tf.reduce_mean(tf.square(y_var))


def get_dataset():
    dataset = tf.data.Dataset.list_files(os.path.join(DATASET_PATH, '*.jpg'))
    dataset = dataset.map(
        load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


def get_content_style_model():

    content_layers = ['block4_conv2']
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    contentstylemodel = ContentStyleModel(content_layers, style_layers)
    return contentstylemodel


@tf.function
def train_step(batch, style_target, network, contentstylemodel, opt, metrics):

    with tf.GradientTape() as tape:

        _, content_target = contentstylemodel(batch*255.0)
        pred = network(batch)
        style_output, content_output = contentstylemodel(pred)
        s_loss = STYLE_WEIGHT * style_loss(style_output, style_target)
        c_loss = CONTENT_WEIGHT * content_loss(content_output, content_target)
        t_loss = TV_WEIGHT * total_variation_loss(pred)
        loss = s_loss + c_loss + t_loss

    gradients = tape.gradient(loss, network.trainable_variables)
    opt.apply_gradients(zip(gradients, network.trainable_variables))

    metrics['style'](s_loss)
    metrics['content'](c_loss)
    metrics['total'](t_loss)
    metrics['loss'](loss)


def train_loop():

    ds = get_dataset()
    contentstylemodel = get_content_style_model()
    network = custom_unet_model()
    network.summary()

    progbar = tf.keras.utils.Progbar(len(ds))

    style_image = load_image(STYLE_IMAGE_PATH, add_dim=True)
    # load images returns normalized images with values between 1 & 1m
    # vgg model expects a scaled up image so we could use is preprocess_input
    style_target, _ = contentstylemodel(style_image*255.0)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001)

    loss_metric = tf.keras.metrics.Mean()
    style_loss_metric = tf.keras.metrics.Mean()
    content_loss_metric = tf.keras.metrics.Mean()
    total_loss_metric = tf.keras.metrics.Mean()

    metrics = {
        'style': style_loss_metric,
        'content': content_loss_metric,
        'total': total_loss_metric,
        'loss': loss_metric
    }

    for e in range(EPOCHS):
        for i, batch in enumerate(ds):
            train_step(batch, style_target, network,
                       contentstylemodel, optimizer, metrics)
            progbar.update(i + 1)
        if e % 1 == 0:
            print(
                f'epoch end: saving weights, style loss--{style_loss_metric.result()}. content loss: {content_loss_metric.result()} tloss: {total_loss_metric.result()}')
            network.save_weights(WEIGHTS_PATH, save_format='tf')
        print(
            f"EPOCH -- {e + 1}: loss--{loss_metric.result()}")
        if e % 2 == 0:
            # validate the image looks good every 10 epoch
            image = load_image(TEST_IMAGE_IMAGE_PATH,
                               add_dim=True, resize=False)
            pred = network(image)
            clipped = clip_image(pred)
            tensor_to_imgarr(clipped).save(PREDICTED_FILE_NAME)
            print('Predicred image')

    network.save_weights(WEIGHTS_PATH, save_format='tf')


train_loop()
