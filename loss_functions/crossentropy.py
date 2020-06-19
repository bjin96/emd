from typing import Callable

import tensorflow.keras.backend as K

from tensorflow.keras.losses import categorical_crossentropy

from models.vgg import EmdWeightHeadStart


def cross_entropy(
        **kwargs
) -> Callable:
    """
    Wrapper for tensorflow.keras.losses.categorical_crossentropy for unified interface with self-guided earth mover
    distance loss.
    """
    return categorical_crossentropy
