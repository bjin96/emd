from typing import Tuple

import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_handlers.adience import get_adience_info, ADIENCE_TRAIN_FOLDS_INFO_FILES, ADIENCE_VALIDATION_FOLDS_INFO_FILES
from data_handlers.preprocessing import paper_preprocessing, paper_preprocessing_validation

AUTOTUNE = tf.data.experimental.AUTOTUNE
X_COLUMN = 'x_col'
Y_COLUMN = 'y_col'


# def get_generators(
#         train_info: pd.DataFrame,
#         validation_info: pd.DataFrame,
# ) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
#     train_generator = ImageDataGenerator(
#         preprocessing_function=paper_preprocessing
#     ).flow_from_dataframe(
#         dataframe=train_info,
#         target_size=(256, 256),
#         x_col=X_COLUMN,
#         y_col=Y_COLUMN
#     )
#     train_generator.class_indices = class_to_index
#     validation_generator = ImageDataGenerator(
#         preprocessing_function=paper_preprocessing_validation
#     ).flow_from_dataframe(
#         dataframe=validation_info,
#         target_size=(256, 256),
#         x_col=X_COLUMN,
#         y_col=Y_COLUMN
#     )
#     validation_generator.class_indices = class_to_index
#     return train_generator, validation_generator


def custom_data_loading(
        train_info: pd.DataFrame,
        validation_info: pd.DataFrame
):
    # batch earlier for performance improvement
    train_dataset = tf.data.Dataset.from_tensor_slices((train_info['x_col'], train_info['y_col']))
    train_dataset = train_dataset.shuffle(100000)
    train_dataset = train_dataset.map(read_images, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.map(paper_preprocessing, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.batch(32)
    train_dataset = train_dataset.prefetch(AUTOTUNE)
    validation_dataset = tf.data.Dataset \
        .from_tensor_slices((validation_info['x_col'], validation_info['y_col'])) \
        .map(read_images, num_parallel_calls=AUTOTUNE) \
        .map(paper_preprocessing_validation, num_parallel_calls=AUTOTUNE) \
        .batch(32) \
        .prefetch(AUTOTUNE)
    return train_dataset, validation_dataset


@tf.function
def read_images(
        file, label
):
    encoded_image = tf.io.read_file(file)
    image = tf.io.decode_jpeg(encoded_image, channels=3)
    return image, tf.one_hot(label, depth=8)


def get_standardized_square_image(image):
    h, w = image.shape[0], image.shape[1]
    if h > w:
        cropped_image = tf.image.crop_to_bounding_box(image, (h - w) // 2, 0, w, w)
    else:
        cropped_image = tf.image.crop_to_bounding_box(image, 0, (w - h) // 2, h, h)
    return tf.image.resize(cropped_image, (256, 256))


def standardize_all_images(train_info, validation_info):
    all_info = pd.concat([train_info, validation_info], axis=0)
    for file in all_info['x_col']:
        image = tf.io.decode_jpeg(tf.io.read_file(file), channels=3)
        standardized_image = get_standardized_square_image(image)
        tf.io.write_file(
            filename=file.replace('faces/', 'test/'),
            contents=tf.io.encode_jpeg(tf.cast(standardized_image, tf.uint8)),
        )


# standardize_all_images(*get_adience_info(ADIENCE_TRAIN_FOLDS_INFO_FILES[0], ADIENCE_VALIDATION_FOLDS_INFO_FILES[0]))
