import pandas as pd
import tensorflow as tf

from data_handlers.preprocessing import paper_preprocessing, paper_preprocessing_validation

AUTOTUNE = tf.data.experimental.AUTOTUNE
X_COLUMN = 'x_col'
Y_COLUMN = 'y_col'


def custom_data_loading(
        train_info: pd.DataFrame,
        validation_info: pd.DataFrame
):
    """
    Load training and validation dataset. Assumes standardized dataset.

    :param train_info: pd.DataFrame containing information about the location of the training dataset.
    :param validation_info: pd.DataFrame containing information about the location of the validation dataset.

    :return: Tuple of tf.data.Dataset for training and validation.
    """
    # Could be batched earlier for performance improvement.
    train_dataset = tf.data.Dataset.from_tensor_slices((train_info['x_col'], train_info['y_col']))
    train_dataset = train_dataset.shuffle(100000)
    train_dataset = train_dataset.map(read_images, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.map(paper_preprocessing, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.batch(32, drop_remainder=True)
    train_dataset = train_dataset.prefetch(AUTOTUNE)
    validation_dataset = tf.data.Dataset \
        .from_tensor_slices((validation_info['x_col'], validation_info['y_col'])) \
        .map(read_images, num_parallel_calls=AUTOTUNE) \
        .map(paper_preprocessing_validation, num_parallel_calls=AUTOTUNE) \
        .batch(32, drop_remainder=True) \
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
    """Standardizes images specified by the arguments."""
    all_info = pd.concat([train_info, validation_info], axis=0)
    for file in all_info['x_col']:
        image = tf.io.decode_jpeg(tf.io.read_file(file), channels=3)
        standardized_image = get_standardized_square_image(image)
        tf.io.write_file(
            filename=file.replace('faces/', 'test/'),
            contents=tf.io.encode_jpeg(tf.cast(standardized_image, tf.uint8)),
        )
