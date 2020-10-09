from typing import Callable, List

import numpy as np
import tensorflow as tf


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
    def _crop_to_central_image(
            image: np.array
    ) -> np.array:
        return tf.image.central_crop(
            image=image,
            central_fraction=central_fraction
        )

    return _crop_to_central_image
