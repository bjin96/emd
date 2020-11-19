from typing import Callable, List

import numpy as np
import tensorflow as tf
from PIL import Image
import PIL

from tensorflow.keras.callbacks import Callback
from tensorflow.python.keras.models import Model


def process_custom_preprocessing(
        preprocessing_functions: List[Callable[[np.array], np.array]]
) -> Callable[[np.array], np.array]:
    """
    Function for aggregating multiple custom preprocessing functions for a
    tf.keras.preprocessing.image.ImageDataGenerator.

    :param preprocessing_functions: List of custom preprocessing functions.

    :return: Aggregate preprocessing function processing all functionality from the passed functions.
    """
    def _process_custom_preprocessing(
            image: np.array
    ) -> np.array:
        processed_image = image.copy()
        for preprocessing_function in preprocessing_functions:
            processed_image = preprocessing_function(processed_image)
        return processed_image

    return _process_custom_preprocessing


def paper_preprocessing(img):
    # crop
    icut = 256 - 227
    jcut = 256 - 227
    ioff = np.random.randint(0, icut + 1)
    joff = np.random.randint(0, jcut + 1)
    img = img[ioff: ioff + img.shape[0] - icut, joff: joff + img.shape[1] - jcut]

    # adjust color
    adj_range = 0.15
    rgb_mean = np.mean(img, axis=(0, 1), keepdims=True).astype(np.float32)
    adj_magn = np.random.uniform(1 - adj_range, 1 + adj_range, (1, 1, 3)).astype(np.float32)
    img = np.clip((img - rgb_mean) * adj_magn + rgb_mean + np.random.uniform(-1.0, 1.0, (1, 1, 3)) * 20, 0.0, 255.0)

    # mirror
    if np.random.rand(1)[0] < 0.5:
        img = img[:, ::-1]

    # scaling
    if np.random.rand(1)[0] < 0.5:
        iscale = 2 * (np.random.rand(1)[0] - 0.5) * 0.10 + 1.0
        jscale = 2 * (np.random.rand(1)[0] - 0.5) * 0.10 + 1.0
        img = np.array(
            Image
                .fromarray(img.astype(np.uint8))
                .resize(
                    size=(int(img.shape[0] * iscale), int(img.shape[1] * jscale)),
                    resample=PIL.Image.BICUBIC
                )
        )

    # rotate small degree
    if np.random.rand(1)[0] < 0.9:
        img = rotate_img(img)

    img = zero_centering(img)
    img = (img - 127.0) / 127.0

    return img


def zero_centering(img):
    x0 = (img.shape[0] - 227) // 2
    y0 = (img.shape[1] - 227) // 2
    im = Image.fromarray(img.astype(np.uint8))
    img = np.array(im.crop((x0, y0, x0+227, y0+227))).astype(np.float32)
    return img


def inverse_transform(X):
    return X * 127.0 + 127.0


def rotate_img(x):
    rotate_angle = (np.random.rand(1)[0] - 0.5) * 2 * 20.0
    im = Image.fromarray(x.astype(np.uint8))
    x = np.array(im.rotate(rotate_angle, Image.BICUBIC)).astype(np.float32)
    return x


def paper_preprocessing_validation(img):
    # crop
    icut = 256 - 227
    jcut = 256 - 227
    ioff = int(icut // 2)
    joff = int(jcut // 2)
    img = img[ioff: ioff + img.shape[0] - icut, joff: joff + img.shape[1] - jcut]

    img = zero_centering(img)
    img = (img - 127.0) / 127.0

    return img


def adjust_aspect_ratio(
        aspect_ratio_range: float
) -> Callable[[np.array], np.array]:
    """
    Returns a function that adjusts the aspect ration of an image by a percentage specified with aspect_ratio_range.

    :param aspect_ratio_range: Maximum amount of adjustment for the aspect ratio change in percent. The applied aspect
    ratio change for a specific image can be negative as well as positive.

    :return: A function that adjusts the input image to the new aspect ratio and pads the output to the original shape.
    """
    def _adjust_aspect_ratio(
            image: np.array
    ) -> np.array:
        if image.ndim != 3:
            raise ValueError('Input image must have rank 3.')
        if image.shape[0] != image.shape[1]:
            raise ValueError('Input image must be squared.')

        aspect_ratio = 1. + (np.random.uniform(-aspect_ratio_range, aspect_ratio_range) / 100.)
        if aspect_ratio < 1:
            target_width = int(image.shape[1] * aspect_ratio)
            target_height = image.shape[0]
        else:
            target_width = image.shape[1]
            target_height = int(image.shape[0] / aspect_ratio)
        mask = np.pad(
            array=np.ones((target_height, target_width, image.shape[2])),
            pad_width=[
                (int(np.ceil((image.shape[0] - target_height) / 2)), (image.shape[0] - target_height) // 2),
                (int(np.ceil((image.shape[1] - target_width) / 2)), (image.shape[1] - target_width) // 2),
                (0, 0)
            ],
            constant_values=0
        )
        return np.where(mask, image, 0)

    return _adjust_aspect_ratio


def crop_to_central_image(
        central_fraction: float
) -> Callable[[np.array], np.array]:
    """
    Returns a function that crops an image to the specified fraction of the original image. The crop is taken from the
    center of the original image.

    :param central_fraction: Fraction of the image to which the image should be cropped.

    :return: A function that crops the image to the desired fraction of the original image.
    """
    central_fraction = central_fraction / 100.

    def _crop_to_central_image(
            image: np.array
    ) -> np.array:
        return tf.image.central_crop(
            image=image,
            central_fraction=central_fraction
        )

    return _crop_to_central_image


def rescale_image(
        target_size: List
) -> Callable[[np.array], np.array]:
    """
    Returns a function that rescales an image to the specified target resolution.

    :param target_size: Target resolution.

    :return: Rescaled image.
    """
    def _rescale_image(
            image: np.array
    ) -> np.array:
        return tf.image.resize(
            images=image,
            size=target_size
        )

    return _rescale_image
