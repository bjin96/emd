from tensorflow.keras.losses import categorical_crossentropy

from evaluation.adience import evaluate_adience_vgg_f, evaluate_adience_alxs, evaluate_adience_res
import tensorflow as tf


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
    evaluate_adience_vgg_f(categorical_crossentropy)
    evaluate_adience_alxs(categorical_crossentropy)
    evaluate_adience_res(categorical_crossentropy)


if __name__ == "__main__":
    main()
