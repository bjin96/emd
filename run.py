from pathlib import Path

import tensorflow as tf
from tensorflow.keras.activations import softmax
from tensorflow.keras.activations import linear

from evaluation.adience import evaluate_adience_model

from loss_functions.crossentropy import cross_entropy
from loss_functions.emd import earth_mover_distance, self_guided_earth_mover_distance, \
    approximate_earth_mover_distance, GroundDistanceManager
from loss_functions.regression import l2_regression_loss
from models.alx import Alxs
from models.res import Res
from models.vgg import Vggf

devices = tf.config.experimental.list_physical_devices('GPU')
if devices:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]
        )
    except RuntimeError as e:
        print(e)


class AdienceXe:

    @staticmethod
    def vggf(lr_index, fold_index):
        evaluate_adience_model(Vggf, lr_index, fold_index, cross_entropy, softmax)

    @staticmethod
    def res(lr_index, fold_index):
        evaluate_adience_model(Res, lr_index, fold_index, cross_entropy, softmax)

    @staticmethod
    def alxs(lr_index, fold_index):
        evaluate_adience_model(Alxs, lr_index, fold_index, cross_entropy, softmax)


class AdienceReg:

    @staticmethod
    def vggf(lr_index, fold_index):
        evaluate_adience_model(Vggf, lr_index, fold_index, l2_regression_loss, linear)

    @staticmethod
    def res(lr_index, fold_index):
        evaluate_adience_model(Res, lr_index, fold_index, l2_regression_loss, linear)

    @staticmethod
    def alxs(lr_index, fold_index):
        evaluate_adience_model(Alxs, lr_index, fold_index, l2_regression_loss, linear)


class AdienceEmd:

    @staticmethod
    def vggf(lr_index, fold_index):
        evaluate_adience_model(Vggf, lr_index, fold_index, earth_mover_distance, softmax)

    @staticmethod
    def res(lr_index, fold_index):
        evaluate_adience_model(Res, lr_index, fold_index, earth_mover_distance, softmax)

    @staticmethod
    def alxs(lr_index, fold_index):
        evaluate_adience_model(
            evaluation_model=Alxs,
            learning_rate_index=lr_index,
            fold_index=fold_index,
            loss_function=earth_mover_distance,
            final_activation=softmax
        )


class AdienceXemd1:

    @staticmethod
    def vggf(lr_index, fold_index):
        evaluate_adience_model(
            evaluation_model=Vggf,
            learning_rate_index=lr_index,
            fold_index=fold_index,
            loss_function=self_guided_earth_mover_distance,
            final_activation=softmax,
            ground_distance_sensitivity=1,
            ground_distance_bias=0.5,
            ground_distance_path=Path(f'ground_distance/vggf/lr={lr_index}_fold={fold_index}')
        )

    @staticmethod
    def res(lr_index, fold_index):
        evaluate_adience_model(
            evaluation_model=Res,
            learning_rate_index=lr_index,
            fold_index=fold_index,
            loss_function=self_guided_earth_mover_distance,
            final_activation=softmax,
            ground_distance_sensitivity=1,
            ground_distance_bias=0.5,
            ground_distance_path=Path(f'ground_distance/res/lr={lr_index}_fold={fold_index}')
        )

    @staticmethod
    def alxs(lr_index, fold_index):
        evaluate_adience_model(
            evaluation_model=Alxs,
            learning_rate_index=lr_index,
            fold_index=fold_index,
            loss_function=self_guided_earth_mover_distance,
            final_activation=softmax,
            ground_distance_sensitivity=1,
            ground_distance_bias=0.5,
            ground_distance_path=Path(f'ground_distance/alxs/lr={lr_index}_fold={fold_index}')
        )


class AdienceXemd2:

    @staticmethod
    def vggf(lr_index, fold_index):
        evaluate_adience_model(
            evaluation_model=Vggf,
            learning_rate_index=lr_index,
            fold_index=fold_index,
            loss_function=self_guided_earth_mover_distance,
            final_activation=softmax,
            ground_distance_sensitivity=2,
            ground_distance_bias=0.25,
            ground_distance_path=Path(f'ground_distance/vggf/lr={lr_index}_fold={fold_index}')
        )

    @staticmethod
    def res(lr_index, fold_index):
        evaluate_adience_model(
            evaluation_model=Res,
            learning_rate_index=lr_index,
            fold_index=fold_index,
            loss_function=self_guided_earth_mover_distance,
            final_activation=softmax,
            ground_distance_sensitivity=2,
            ground_distance_bias=0.25,
            ground_distance_path=Path(f'ground_distance/res/lr={lr_index}_fold={fold_index}')
        )

    @staticmethod
    def alxs(lr_index, fold_index):
        evaluate_adience_model(
            evaluation_model=Alxs,
            learning_rate_index=lr_index,
            fold_index=fold_index,
            loss_function=self_guided_earth_mover_distance,
            final_activation=softmax,
            ground_distance_sensitivity=2,
            ground_distance_bias=0.25,
            ground_distance_path=Path(f'ground_distance/alxs/lr={lr_index}_fold={fold_index}')
        )


class AdienceAemd1:
    """Entropic regularizer = 0.1"""

    @staticmethod
    def vggf(lr_index, fold_index):
        ground_distance_manager = GroundDistanceManager(Path('ground_distances'))
        ground_distance_matrix = ground_distance_manager.load_ground_distance_matrix('159')
        evaluate_adience_model(
            evaluation_model=Vggf,
            learning_rate_index=lr_index,
            fold_index=fold_index,
            loss_function=approximate_earth_mover_distance,
            final_activation=softmax,
            entropic_regularizer=0.1,
            distance_matrix=ground_distance_matrix,
            matrix_scaling_operations=100
        )

    @staticmethod
    def res(lr_index, fold_index):
        ground_distance_manager = GroundDistanceManager(Path('ground_distances'))
        ground_distance_matrix = ground_distance_manager.load_ground_distance_matrix('159')
        evaluate_adience_model(
            evaluation_model=Res,
            learning_rate_index=lr_index,
            fold_index=fold_index,
            loss_function=approximate_earth_mover_distance,
            final_activation=softmax,
            entropic_regularizer=0.1,
            distance_matrix=ground_distance_matrix,
            matrix_scaling_operations=100
        )

    @staticmethod
    def alxs(lr_index, fold_index):
        ground_distance_manager = GroundDistanceManager(Path('ground_distances'))
        ground_distance_matrix = ground_distance_manager.load_ground_distance_matrix('159')
        evaluate_adience_model(
            evaluation_model=Alxs,
            learning_rate_index=lr_index,
            fold_index=fold_index,
            loss_function=approximate_earth_mover_distance,
            final_activation=softmax,
            entropic_regularizer=0.1,
            distance_matrix=ground_distance_matrix,
            matrix_scaling_operations=100
        )


class AdienceAemd2:
    """Entropic regularizer = 1"""

    @staticmethod
    def vggf(lr_index, fold_index):
        ground_distance_manager = GroundDistanceManager(Path('ground_distances'))
        ground_distance_matrix = ground_distance_manager.load_ground_distance_matrix('159')
        evaluate_adience_model(
            evaluation_model=Vggf,
            learning_rate_index=lr_index,
            fold_index=fold_index,
            loss_function=approximate_earth_mover_distance,
            final_activation=softmax,
            entropic_regularizer=1,
            distance_matrix=ground_distance_matrix,
            matrix_scaling_operations=100
        )

    @staticmethod
    def res(lr_index, fold_index):
        ground_distance_manager = GroundDistanceManager(Path('ground_distances'))
        ground_distance_matrix = ground_distance_manager.load_ground_distance_matrix('159')
        evaluate_adience_model(
            evaluation_model=Res,
            learning_rate_index=lr_index,
            fold_index=fold_index,
            loss_function=approximate_earth_mover_distance,
            final_activation=softmax,
            entropic_regularizer=1,
            distance_matrix=ground_distance_matrix,
            matrix_scaling_operations=100
        )

    @staticmethod
    def alxs(lr_index, fold_index):
        ground_distance_manager = GroundDistanceManager(Path('ground_distances'))
        ground_distance_matrix = ground_distance_manager.load_ground_distance_matrix('159')
        evaluate_adience_model(
            evaluation_model=Alxs,
            learning_rate_index=lr_index,
            fold_index=fold_index,
            loss_function=approximate_earth_mover_distance,
            final_activation=softmax,
            entropic_regularizer=1,
            distance_matrix=ground_distance_matrix,
            matrix_scaling_operations=100
        )


class AdienceAemd3:
    """Entropic regularizer = 10"""

    @staticmethod
    def vggf(lr_index, fold_index):
        ground_distance_manager = GroundDistanceManager(Path('ground_distances'))
        ground_distance_matrix = ground_distance_manager.load_ground_distance_matrix('159')
        evaluate_adience_model(
            evaluation_model=Vggf,
            learning_rate_index=lr_index,
            fold_index=fold_index,
            loss_function=approximate_earth_mover_distance,
            final_activation=softmax,
            entropic_regularizer=10,
            distance_matrix=ground_distance_matrix,
            matrix_scaling_operations=100
        )

    @staticmethod
    def res(lr_index, fold_index):
        ground_distance_manager = GroundDistanceManager(Path('ground_distances'))
        ground_distance_matrix = ground_distance_manager.load_ground_distance_matrix('159')
        evaluate_adience_model(
            evaluation_model=Res,
            learning_rate_index=lr_index,
            fold_index=fold_index,
            loss_function=approximate_earth_mover_distance,
            final_activation=softmax,
            entropic_regularizer=10,
            distance_matrix=ground_distance_matrix,
            matrix_scaling_operations=100
        )

    @staticmethod
    def alxs(lr_index, fold_index):
        ground_distance_manager = GroundDistanceManager(Path('ground_distances'))
        ground_distance_matrix = ground_distance_manager.load_ground_distance_matrix('159')
        evaluate_adience_model(
            evaluation_model=Alxs,
            learning_rate_index=lr_index,
            fold_index=fold_index,
            loss_function=approximate_earth_mover_distance,
            final_activation=softmax,
            entropic_regularizer=10,
            distance_matrix=ground_distance_matrix,
            matrix_scaling_operations=100
            )
