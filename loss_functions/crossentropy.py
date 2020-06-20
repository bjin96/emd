from typing import Callable

from tensorflow.keras.losses import categorical_crossentropy


def cross_entropy(
        **kwargs
) -> Callable:
    """
    Wrapper for tensorflow.keras.losses.categorical_crossentropy for unified interface with self-guided earth mover
    distance loss.
    """
    return categorical_crossentropy
