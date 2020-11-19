from typing import Tuple

import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_handlers.preprocessing import paper_preprocessing, paper_preprocessing_validation

X_COLUMN = 'x_col'
Y_COLUMN = 'y_col'


def get_generators(
        train_info: pd.DataFrame,
        validation_info: pd.DataFrame,
) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
    train_generator = ImageDataGenerator(
        preprocessing_function=paper_preprocessing
    ).flow_from_dataframe(
        dataframe=train_info,
        target_size=(227, 227),
        x_col=X_COLUMN,
        y_col=Y_COLUMN
    )
    validation_generator = ImageDataGenerator(
        preprocessing_function=paper_preprocessing_validation
    ).flow_from_dataframe(
        dataframe=validation_info,
        target_size=(227, 227),
        x_col=X_COLUMN,
        y_col=Y_COLUMN
    )
    return train_generator, validation_generator
