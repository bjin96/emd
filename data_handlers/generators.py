from typing import Tuple

import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_handlers.preprocessing import adjust_aspect_ratio, process_custom_preprocessing, crop_to_central_image, \
    rescale_image

X_COLUMN = 'x_col'
Y_COLUMN = 'y_col'


def get_generators(
        train_info: pd.DataFrame,
        validation_info: pd.DataFrame,
) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
    train_generator = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        channel_shift_range=10.,
        preprocessing_function=process_custom_preprocessing(
            preprocessing_functions=[
                rescale_image([255, 255]),
                crop_to_central_image(89.),
                adjust_aspect_ratio(10.)
            ]
        )
    ).flow_from_dataframe(
        dataframe=train_info,
        target_size=(227, 227),
        x_col=X_COLUMN,
        y_col=Y_COLUMN
    )
    validation_generator = ImageDataGenerator(
        horizontal_flip=True,
        preprocessing_function=process_custom_preprocessing(
            preprocessing_functions=[
                rescale_image([255, 255]),
                crop_to_central_image(89.)
            ]
        )
    ).flow_from_dataframe(
        dataframe=validation_info,
        target_size=(227, 227),
        x_col=X_COLUMN,
        y_col=Y_COLUMN
    )
    return train_generator, validation_generator
