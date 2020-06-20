from typing import Callable

from tensorflow.keras.losses import mean_squared_error


def l2_regression_loss(
        **kwargs
) -> Callable:
    """
    Wrapper for tensorflow.keras.losses.mean_square_error for unified interface with self-guided earth mover
    distance loss.
    """
    return mean_squared_error
