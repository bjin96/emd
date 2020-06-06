from tensorflow.keras.activations import softmax
from tensorflow.keras.activations import linear
from tensorflow.keras.losses import categorical_crossentropy, mean_squared_error

from evaluation.adience import evaluate_adience_vgg_f, evaluate_adience_alxs, evaluate_adience_res
import tensorflow as tf

from loss_functions.emd import earth_mover_distance

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
    evaluate_adience_vgg_f(categorical_crossentropy, softmax)
    evaluate_adience_alxs(categorical_crossentropy, softmax)
    evaluate_adience_res(categorical_crossentropy, softmax)

    # Evaluate with L2 regression loss:
    evaluate_adience_vgg_f(earth_mover_distance, softmax)
    evaluate_adience_alxs(earth_mover_distance, softmax)
    evaluate_adience_res(earth_mover_distance, softmax)

    # Evaluate with EMD^2 loss:
    evaluate_adience_vgg_f(mean_squared_error, linear)
    evaluate_adience_alxs(mean_squared_error, linear)
    evaluate_adience_res(mean_squared_error, linear)


if __name__ == "__main__":
    main()
