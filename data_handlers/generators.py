from typing import Tuple

import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_handlers.preprocessing import paper_preprocessing, paper_preprocessing_validation

X_COLUMN = 'x_col'
Y_COLUMN = 'y_col'
class_to_index = {
    '(0, 2)': 0,
    '(4, 6)': 1,
    '(8, 12)': 2,
    '(15, 20)': 3,
    '(25, 32)': 4,
    '(38, 43)': 5,
    '(48, 53)': 6,
    '(60, 100)': 7
}

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
    train_generator.class_indices = class_to_index
    validation_generator = ImageDataGenerator(
        preprocessing_function=paper_preprocessing_validation
    ).flow_from_dataframe(
        dataframe=validation_info,
        target_size=(227, 227),
        x_col=X_COLUMN,
        y_col=Y_COLUMN
    )
    validation_generator.class_indices = class_to_index
    return train_generator, validation_generator
