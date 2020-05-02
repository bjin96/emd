from typing import Tuple

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

X_COLUMN = 'x_col'
Y_COLUMN = 'y_col'


def get_generators(
        train_info: pd.DataFrame,
        validation_info: pd.DataFrame,
) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
    # no adjustment of aspect ratio
    generator = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        channel_shift_range=10.
    )
    train_generator = generator.flow_from_dataframe(
        dataframe=train_info,
        x_col=X_COLUMN,
        y_col=Y_COLUMN
    )
    validation_generator = generator.flow_from_dataframe(
        dataframe=validation_info,
        x_col=X_COLUMN,
        y_col=Y_COLUMN
    )
    return train_generator, validation_generator
