import fire
from tensorflow.keras.activations import softmax
from tensorflow.keras.activations import linear

from evaluation.adience import evaluate_adience_model
import tensorflow as tf

from loss_functions.crossentropy import cross_entropy
from loss_functions.emd import earth_mover_distance, self_guided_earth_mover_distance, approximate_earth_mover_distance, \
    load_ground_distance_matrix
from loss_functions.regression import l2_regression_loss
from models.alx import Alxs
from models.res import Res
from models.vgg import Vggf

devices = tf.config.experimental.list_physical_devices('GPU')
if devices:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
    except RuntimeError as e:
        print(e)


class AdienceXe:

    def vggf(self):
        evaluate_adience_model(Vggf, cross_entropy, softmax)

    def res(self):
        evaluate_adience_model(Res, cross_entropy, softmax)

    def alxs(self):
        evaluate_adience_model(Alxs, cross_entropy, softmax)


class AdienceReg:

    def vggf(self):
        evaluate_adience_model(Vggf, l2_regression_loss, linear)

    def res(self):
        evaluate_adience_model(Res, l2_regression_loss, linear)

    def alxs(self):
        evaluate_adience_model(Alxs, l2_regression_loss, linear)


class AdienceEmd:

    def vggf(self):
        evaluate_adience_model(Vggf, earth_mover_distance, softmax)

    def res(self):
        evaluate_adience_model(Res, earth_mover_distance, softmax)

    def alxs(self):
        evaluate_adience_model(Alxs, earth_mover_distance, softmax)


class AdienceXemd1:

    def vggf(self):
        evaluate_adience_model(
            evaluation_model=Vggf,
            loss_function=self_guided_earth_mover_distance,
            final_activation=softmax,
            ground_distance_sensitivity=1,
            ground_distance_bias=0.5
        )

    def res(self):
        evaluate_adience_model(
            evaluation_model=Res,
            loss_function=self_guided_earth_mover_distance,
            final_activation=softmax,
            ground_distance_sensitivity=1,
            ground_distance_bias=0.5
        )

    def alxs(self):
        evaluate_adience_model(
            evaluation_model=Alxs,
            loss_function=self_guided_earth_mover_distance,
            final_activation=softmax,
            ground_distance_sensitivity=1,
            ground_distance_bias=0.5
        )


class AdienceXemd2:

    def vggf(self):
        evaluate_adience_model(
            evaluation_model=Vggf,
            loss_function=self_guided_earth_mover_distance,
            final_activation=softmax,
            ground_distance_sensitivity=2,
            ground_distance_bias=0.25
        )

    def res(self):
        evaluate_adience_model(
            evaluation_model=Res,
            loss_function=self_guided_earth_mover_distance,
            final_activation=softmax,
            ground_distance_sensitivity=2,
            ground_distance_bias=0.25
        )

    def alxs(self):
        evaluate_adience_model(
            evaluation_model=Alxs,
            loss_function=self_guided_earth_mover_distance,
            final_activation=softmax,
            ground_distance_sensitivity=2,
            ground_distance_bias=0.25
        )


class AdienceAemd1:
    """Entropic regularizer = 0.1"""

    def vggf(self):
        ground_distance_matrix = load_ground_distance_matrix()
        evaluate_adience_model(
            evaluation_model=Vggf,
            loss_function=approximate_earth_mover_distance,
            final_activation=softmax,
            entropic_regularizer=0.1,
            distance_matrix=ground_distance_matrix,
            matrix_scaling_operations=100
        )

    def res(self):
        ground_distance_matrix = load_ground_distance_matrix()
        evaluate_adience_model(
            evaluation_model=Res,
            loss_function=approximate_earth_mover_distance,
            final_activation=softmax,
            entropic_regularizer=0.1,
            distance_matrix=ground_distance_matrix,
            matrix_scaling_operations=100
        )

    def alxs(self):
        ground_distance_matrix = load_ground_distance_matrix()
        evaluate_adience_model(
            evaluation_model=Alxs,
            loss_function=approximate_earth_mover_distance,
            final_activation=softmax,
            entropic_regularizer=0.1,
            distance_matrix=ground_distance_matrix,
            matrix_scaling_operations=100
        )


class AdienceAemd2:
    """Entropic regularizer = 1"""

    def vggf(self):
        ground_distance_matrix = load_ground_distance_matrix()
        evaluate_adience_model(
            evaluation_model=Vggf,
            loss_function=approximate_earth_mover_distance,
            final_activation=softmax,
            entropic_regularizer=1,
            distance_matrix=ground_distance_matrix,
            matrix_scaling_operations=100
        )

    def res(self):
        ground_distance_matrix = load_ground_distance_matrix()
        evaluate_adience_model(
            evaluation_model=Res,
            loss_function=approximate_earth_mover_distance,
            final_activation=softmax,
            entropic_regularizer=1,
            distance_matrix=ground_distance_matrix,
            matrix_scaling_operations=100
        )

    def alxs(self):
        ground_distance_matrix = load_ground_distance_matrix()
        evaluate_adience_model(
            evaluation_model=Alxs,
            loss_function=approximate_earth_mover_distance,
            final_activation=softmax,
            entropic_regularizer=1,
            distance_matrix=ground_distance_matrix,
            matrix_scaling_operations=100
        )


class AdienceAemd3:
    """Entropic regularizer = 10"""

    def vggf(self):
        ground_distance_matrix = load_ground_distance_matrix()
        evaluate_adience_model(
            evaluation_model=Vggf,
            loss_function=approximate_earth_mover_distance,
            final_activation=softmax,
            entropic_regularizer=10,
            distance_matrix=ground_distance_matrix,
            matrix_scaling_operations=100
        )

    def res(self):
        ground_distance_matrix = load_ground_distance_matrix()
        evaluate_adience_model(
            evaluation_model=Res,
            loss_function=approximate_earth_mover_distance,
            final_activation=softmax,
            entropic_regularizer=10,
            distance_matrix=ground_distance_matrix,
            matrix_scaling_operations=100
        )

    def alxs(self):
        ground_distance_matrix = load_ground_distance_matrix()
        evaluate_adience_model(
            evaluation_model=Alxs,
            loss_function=approximate_earth_mover_distance,
            final_activation=softmax,
            entropic_regularizer=10,
            distance_matrix=ground_distance_matrix,
            matrix_scaling_operations=100
            )


if __name__ == "__main__":
    fire.Fire()
