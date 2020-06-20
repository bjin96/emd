from tensorflow.keras.activations import softmax
from tensorflow.keras.activations import linear
from tensorflow.keras.losses import categorical_crossentropy, mean_squared_error

from evaluation.adience import evaluate_adience_model
import tensorflow as tf

from loss_functions.emd import earth_mover_distance, self_guided_earth_mover_distance
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
    evaluate_adience_model(Vggf, categorical_crossentropy, softmax)
    evaluate_adience_model(Resf, categorical_crossentropy, softmax)
    evaluate_adience_model(Alxs, categorical_crossentropy, softmax)

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
    evaluate_adience_model(Vggf, mean_squared_error, linear)
    evaluate_adience_model(Resf, mean_squared_error, linear)
    evaluate_adience_model(Alxs, mean_squared_error, linear)


if __name__ == "__main__":
    main()
