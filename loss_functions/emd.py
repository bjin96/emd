import os
from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import categorical_crossentropy

GROUND_DISTANCE_FILE = os.path.dirname(__file__) / Path('../ground_distance.npy')


def earth_mover_distance(
        **kwargs
) -> Callable:
    """
    Wrapper for earth_mover distance for unified interface with self-guided earth mover distance loss.
    """
    def _earth_mover_distance(
            y_true: K.placeholder,
            y_pred: K.placeholder
    ) -> K.placeholder:
        return tf.reduce_mean(tf.square(tf.cumsum(y_true, axis=-1) - tf.cumsum(y_pred, axis=-1)), axis=-1)

    return _earth_mover_distance


def approximate_earth_mover_distance(
        entropic_regularizer: float,
        distance_matrix: np.array,
        matrix_scaling_operations: int = 100,
        **kwargs
) -> Callable:
    """
    Wrapper for approximate earth mover distance.
    """

    def _approximate_earth_mover_distance(
            y_true: K.placeholder,
            y_pred: K.placeholder
    ) -> K.placeholder:
        k = tf.exp(-entropic_regularizer * distance_matrix)
        km = k * distance_matrix
        u = tf.ones(y_true.shape) / y_true.shape[1]
        for _ in range(matrix_scaling_operations):
            u = y_pred / ((y_true / (u @ k)) @ k)
        v = y_true / (u @ k)
        return tf.reduce_sum(u * (v @ km)) / y_true.shape[0]

    return _approximate_earth_mover_distance


class EmdWeightHeadStart(Callback):

    def __init__(self):
        super(EmdWeightHeadStart, self).__init__()
        self.emd_weight = False

    def on_epoch_begin(self, epoch, logs={}):
        if epoch == 5:
            self.emd_weight = True


def self_guided_earth_mover_distance(
        model,
        ground_distance_sensitivity: float,
        ground_distance_bias: float
) -> Callable:

    def _self_guided_earth_mover_distance(
            y_true: K.placeholder,
            y_pred: K.placeholder
    ) -> K.placeholder:
        class_features = model.second_to_last_layer
        cross_entropy_loss = categorical_crossentropy(
            y_true=y_true,
            y_pred=y_pred
        )
        if model.emd_weight_head_start.emd_weight:
            self_guided_emd_loss = _calculate_self_guided_loss(
                y_true=y_true,
                y_pred=y_pred,
                ground_distance_sensitivity=ground_distance_sensitivity,
                ground_distance_bias=ground_distance_bias,
                class_features=class_features
            )
            loss_function_relation = (cross_entropy_loss / self_guided_emd_loss) / 3.5
            return cross_entropy_loss \
                + model.emd_weight_head_start.emd_weight * loss_function_relation * self_guided_emd_loss
        else:
            return cross_entropy_loss

    return _self_guided_earth_mover_distance


def _calculate_self_guided_loss(
        y_true: K.placeholder,
        y_pred: K.placeholder,
        ground_distance_sensitivity: float,
        ground_distance_bias: float,
        class_features: K.placeholder
):
    class_length = 8
    batch_size = 32
    estimated_distances = _estimate_distances(
        class_features=class_features,
        y_true=y_true,
        class_length=class_length
    )
    ground_distances = _calculate_ground_distances(
        estimated_distances=estimated_distances,
        class_length=class_length
    )
    save_ground_distance_matrix(ground_distances)
    cost_vectors = []
    for i in range(batch_size):
        cost_vectors.append(
            ground_distances[:, K.argmax(y_true[i])] ** ground_distance_sensitivity + ground_distance_bias
        )
    cost_vectors = tf.stack(cost_vectors)
    return K.sum(K.square(y_pred) * cost_vectors, axis=1)


def _estimate_distances(
        class_features: K.placeholder,
        y_true: K.placeholder,
        class_length: int
) -> K.placeholder:
    sample_means = K.mean(class_features, axis=-1)
    class_labels = K.argmax(y_true, axis=-1)
    centroids = []
    for i in range(class_length):
        centroids.append(K.mean(sample_means[class_labels == i]))
    centroids = tf.stack(centroids)
    # In case a class does not appear in the current batch, use centroid mean as centroid for that class
    centroids = tf.where(
        condition=tf.math.is_nan(centroids),
        x=tf.reduce_mean(centroids[tf.logical_not(tf.math.is_nan(centroids))]),
        y=centroids
    )
    estimated_distances = []
    for i in range(class_length):
        estimated_distances.append(K.square(centroids[i] - centroids))
    return tf.stack(estimated_distances)


def _calculate_ground_distances(
        estimated_distances: K.placeholder,
        class_length: int
) -> K.placeholder:
    sorted_indices = tf.argsort(estimated_distances)
    elements_smaller = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            elements_smaller[i, sorted_indices[i, j]] = j
    elements_smaller = tf.convert_to_tensor(elements_smaller, dtype=tf.float32)
    normalized_distances = (1 / class_length) * elements_smaller
    return (normalized_distances + K.transpose(normalized_distances)) / 2


def save_ground_distance_matrix(
        ground_distance_matrix: K.placeholder
) -> None:
    np.save(
        file=GROUND_DISTANCE_FILE,
        arr=ground_distance_matrix
    )


def load_ground_distance_matrix() -> np.array:
    return np.load(GROUND_DISTANCE_FILE)
