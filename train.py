import os
import tensorflow as tf
import numpy as np
from utils import load_image

from model import ContentStyleModel, custom_unet_model

DATASET_PATH = os.path.join(os.getcwd(), 'dataset', 'val2017')
STYLE_IMAGE_PATH = os.path.join(os.getcwd(), 'dataset', 'style.jpg')
WEIGHTS_PATH = os.path.join(os.getcwd(), 'weights')
TRAIN_INPUT_SHAPE = (224, 224, 3)
BUFFER_SIZE = 100
BATCH_SIZE = 16
EPOCHS = 1
CONTENT_WEIGHT = 6e0
STYLE_WEIGHT = 2e-3
TV_WEIGHT = 6e2

def style_loss(style_output, style_target):

    style_loss = tf.add_n(tf.reduce_mean(tf.square(style_output[i] - style_target[i])) for i in range(style_target))
    return style_loss / len(style_output)

def content_loss(content_output, content_target):

    content_loss = tf.add_n(tf.reduce_mean(tf.square(content_output[i] - content_target[i])) for i in range(content_target))
    return content_loss / len(content_output)

def total_variation_loss(img):
    x_var = img[:,:,1:,:] - img[:,:,:-1,:]
    y_var = img[:,1:,:,:] - img[:,:-1,:,:]
    return tf.reduce_mean(tf.square(x_var) + tf.reduce_mean(tf.square(y_var)))

def get_dataset():
    dataset = tf.data.Dataset.list_files(os.path.join(DATASET_PATH, '*.jpg'))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return dataset


def get_content_style_model():

    content_layers = ['block4_conv3']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    contentstylemodel = ContentStyleModel(content_layers, style_layers)
    return contentstylemodel


@tf.function
def train_step(batch, style_target, network, contentstylemodel, opt, metrics):

    with tf.GradientTape() as tape:

        _, content_target = contentstylemodel(batch)
        pred = network(batch)
        style_output, content_output = contentstylemodel(pred)

        s_loss = STYLE_WEIGHT * style_loss(style_output, style_target)
        c_loss = CONTENT_WEIGHT * content_loss(content_output, content_target)
        t_loss = TV_WEIGHT * total_variation_loss(pred)
        loss = s_loss + c_loss + t_loss

        metrics.style(s_loss)
        metrics.content(c_loss)
        metrics.total(t_loss)
        metrics.loss(loss)

    gradients = tape.gradient(loss, network.trainable_variables)
    opt.apply_gradient(zip(gradients, network.trainable_variables))



def train_loop():

    ds = get_dataset()

    contentstylemodel = get_content_style_model()
    network = custom_unet_model(input_shape=TRAIN_INPUT_SHAPE)

    style_image = load_image(STYLE_IMAGE_PATH, add_dim=True)
    style_target, _ = contentstylemodel(style_image)

    optimizer = tf.keras.optimizers.Adam()

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
            
            train_step(batch, style_target, network, contentstylemodel, optimizer, metrics)

            if i % 1000 == 0:
                network.save_weights(WEIGHTS_PATH)

    
        print(f"EPOCH -- {e}: loss--{loss_metric.result()}")
    network.save_weights(WEIGHTS_PATH)

train_loop()