import os
import numpy as np
from argparse import ArgumentParser
from model import custom_unet_model
from utils import tensor_to_imgarr, load_image, clip_image


WEIGHTS_PATH = os.path.join(os.getcwd(), 'weights')
PREDICTED_FILE_NAME = 'new.jpg'


def get_model_with_weights(weights=WEIGHTS_PATH):
    model = custom_unet_model()
    model.load_weights(weights).expect_partial()
    return model


parser = ArgumentParser(description="Arguments to predict style")
parser.add_argument('--image', '-i', required=True,
                    help="content image to evaluate")


if __name__ == '__main__':
    args = parser.parse_args()

    model = get_model_with_weights()
    print('Model created..')
    image = load_image(args.image, add_dim=True, resize=False)
    print('Image loaded', image.shape)
    pred = model(image)
    print('Predicred...')
    clipped = clip_image(pred)
    print('Clipped...')
    tensor_to_imgarr(clipped).save(PREDICTED_FILE_NAME)
