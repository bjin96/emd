from tensorflow.keras.activations import softmax
from tensorflow.keras.activations import linear

from evaluation.adience import evaluate_adience_model
import tensorflow as tf

from loss_functions.crossentropy import cross_entropy
from loss_functions.emd import earth_mover_distance, self_guided_earth_mover_distance
from loss_functions.regression import l2_regression_loss
from models.alx import Alxs
from models.res import Resf
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


def main():
    # Evaluate with cross-entropy loss:
    evaluate_adience_model(Vggf, cross_entropy, softmax)
    evaluate_adience_model(Resf, cross_entropy, softmax)
    evaluate_adience_model(Alxs, cross_entropy, softmax)

    # Evaluate with EMD^2 loss:
    evaluate_adience_model(Vggf, earth_mover_distance, softmax)
    evaluate_adience_model(Resf, earth_mover_distance, softmax)
    evaluate_adience_model(Alxs, earth_mover_distance, softmax)

    # Evaluate with self-guided EMD^2 loss:
    evaluate_adience_model(
        evaluation_model=Vggf,
        loss_function=self_guided_earth_mover_distance,
        final_activation=softmax,
        ground_distance_sensitivity=1,
        ground_distance_bias=0.5
    )
    evaluate_adience_model(
        evaluation_model=Resf,
        loss_function=self_guided_earth_mover_distance,
        final_activation=softmax,
        ground_distance_sensitivity=1,
        ground_distance_bias=0.5
    )
    evaluate_adience_model(
        evaluation_model=Alxs,
        loss_function=self_guided_earth_mover_distance,
        final_activation=softmax,
        ground_distance_sensitivity=1,
        ground_distance_bias=0.5
    )

    # Evaluate with L2 regression loss:
    evaluate_adience_model(Vggf, l2_regression_loss, linear)
    evaluate_adience_model(Resf, l2_regression_loss, linear)
    evaluate_adience_model(Alxs, l2_regression_loss, linear)


if __name__ == "__main__":
    main()
