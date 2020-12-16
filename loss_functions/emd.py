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
        self.emd_weight = 0

    def on_epoch_begin(self, epoch, logs={}):
        if epoch == 4:
            self.emd_weight = 1


class GroundDistanceManager(Callback):

    def __init__(
            self,
            file_path: Path
    ):
        super(GroundDistanceManager, self).__init__()
        self.ground_distance_matrix = None
        self.epoch_class_features = []
        self.epoch_labels = []
        self.class_length = 8

        file_path.mkdir(parents=True, exist_ok=True)
        self.file_path = file_path

    def set_labels(self, labels):
        labels_tensor = tf.concat(labels, axis=0)
        self.epoch_labels = labels_tensor

    def on_batch_end(self, batch, logs=None):
        self.epoch_class_features.append(self.model.second_to_last_layer)

    def on_epoch_end(self, epoch, logs=None):
        self._update_ground_distance_matrix()
        self._save_ground_distance_matrix(epoch=epoch)

    def _update_ground_distance_matrix(self):
        self.epoch_class_features = tf.concat(self.epoch_class_features, axis=0)
        estimated_distances = self._estimate_distances()
        self.ground_distance_matrix = self._calculate_ground_distances(
            estimated_distances=estimated_distances
        )
        self.epoch_class_features = []
        self.epoch_labels = []

    def _estimate_distances(self) -> K.placeholder:
        normalized_features = tf.norm(
            tensor=self.epoch_class_features,
            ord=1,
            axis=-1
        )
        class_labels = K.argmax(self.epoch_labels, axis=-1)
        centroids = []
        for i in range(self.class_length):
            centroids.append(K.mean(normalized_features[class_labels == i]))
        centroids = tf.stack(centroids)
        estimated_distances = []
        for i in range(self.class_length):
            estimated_distances.append(tf.norm(centroids[i] - centroids))
        return tf.stack(estimated_distances)

    def _calculate_ground_distances(
            self,
            estimated_distances: K.placeholder
    ) -> K.placeholder:
        sorted_indices = tf.argsort(estimated_distances)
        elements_smaller = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                elements_smaller[i, sorted_indices[i, j]] = j
        elements_smaller = tf.convert_to_tensor(elements_smaller, dtype=tf.float32)
        normalized_distances = (1 / self.class_length) * elements_smaller
        return (normalized_distances + K.transpose(normalized_distances)) / 2

    def _save_ground_distance_matrix(self, epoch: int) -> None:
        np.save(
            file=str(self.file_path) + f'/{epoch}',
            arr=self.ground_distance_matrix
        )

    def load_ground_distance_matrix(self) -> np.array:
        return np.load(str(self.file_path))


def self_guided_earth_mover_distance(
        model,
        ground_distance_sensitivity: float,
        ground_distance_bias: float
) -> Callable:

    def _self_guided_earth_mover_distance(
            y_true: K.placeholder,
            y_pred: K.placeholder,
    ) -> K.placeholder:
        cross_entropy_loss = categorical_crossentropy(
            y_true=y_true,
            y_pred=y_pred
        )
        if model.emd_weight_head_start.emd_weight == 0:
            return cross_entropy_loss
        else:
            self_guided_emd_loss = _calculate_self_guided_loss(
                y_true=y_true,
                y_pred=y_pred,
                ground_distance_sensitivity=ground_distance_sensitivity,
                ground_distance_bias=ground_distance_bias,
                ground_distance_manager=model.ground_distance_manager
            )
            # loss_function_relation = (cross_entropy_loss / self_guided_emd_loss) / 3.5
            return cross_entropy_loss \
                + 5.0 * self_guided_emd_loss

    return _self_guided_earth_mover_distance


def _calculate_self_guided_loss(
        y_true: K.placeholder,
        y_pred: K.placeholder,
        ground_distance_sensitivity: float,
        ground_distance_bias: float,
        ground_distance_manager: GroundDistanceManager
):
    batch_size = 32
    cost_vectors = []
    for i in range(batch_size):
        cost_vectors.append(
            ground_distance_manager.ground_distance_matrix[:, K.argmax(y_true[i])] ** ground_distance_sensitivity + ground_distance_bias
        )
    cost_vectors = tf.stack(cost_vectors)
    return K.sum(K.square(y_pred) * cost_vectors, axis=1)



