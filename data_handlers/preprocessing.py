from typing import Callable

import numpy as np


def adjust_aspect_ratio(
        aspect_ratio_range: float
) -> Callable[[np.array], np.array]:
    """
    Returns a function that adjusts the aspect ration of an image by a percentage specified with aspect_ratio_range.

    :param aspect_ratio_range: Maximum amount of adjustment for the aspect ratio change in percent. The applied aspect
    ratio change for a specific image can be negative as well as positive.

    :return: The input image with the new aspect ratio and padded to be of the original shape.
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
