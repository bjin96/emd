import tensorflow as tf


def one_off_accuracy(y_true, y_pred):
    true_indices = tf.argmax(y_true, axis=-1, output_type=tf.dtypes.int32)
    pred_indices = tf.argmax(y_pred, axis=-1, output_type=tf.dtypes.int32)
    return tf.cast(tf.less_equal(tf.abs(true_indices - pred_indices), tf.constant([1])), tf.float32)
