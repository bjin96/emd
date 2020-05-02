from keras import backend as K


def earth_mover_distance(
        y_true: K.placeholder,
        y_pred: K.placeholder
) -> K.placeholder:
    return K.sum(K.square(K.cumsum(y_true, axis=-1) - K.cumsum(y_pred, axis=-1)), axis=-1)
