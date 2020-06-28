import tensorflow as tf


def one_off_accuracy(y_true, y_pred):
    true_indices = tf.argmax(y_true, axis=-1)
    pred_indices = tf.argmax(y_pred, axis=-1)
    return tf.reduce_sum(tf.cast(tf.abs(true_indices - pred_indices) <= 1, tf.int32))
