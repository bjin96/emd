from typing import Callable

import tensorflow as tf
import numpy as np

from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import backend as K

from models.vgg import EmdWeightHeadStart


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
        return K.sum(K.square(K.cumsum(y_true, axis=-1) - K.cumsum(y_pred, axis=-1)), axis=-1)

    return _earth_mover_distance


def self_guided_earth_mover_distance(
        second_to_last: K.placeholder,
        emd_weight_head_start: EmdWeightHeadStart,
        ground_distance_sensitivity: float,
        ground_distance_bias: float
) -> Callable:

    def _self_guided_earth_mover_distance(
            y_true: K.placeholder,
            y_pred: K.placeholder
    ) -> K.placeholder:
        class_features = second_to_last
        cross_entropy_loss = categorical_crossentropy(
            y_true=y_true,
            y_pred=y_pred
        )
        if emd_weight_head_start.emd_weight:
            self_guided_emd_loss = _calculate_self_guided_loss(
                y_true=y_true,
                y_pred=y_pred,
                ground_distance_sensitivity=ground_distance_sensitivity,
                ground_distance_bias=ground_distance_bias,
                class_features=class_features
            )
            loss_function_relation = (cross_entropy_loss / self_guided_emd_loss) / 3.5
            return cross_entropy_loss + emd_weight_head_start.emd_weight * loss_function_relation * self_guided_emd_loss
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
