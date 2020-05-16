from typing import Callable

import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Layer
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.optimizers import SGD

from models.constants import OPTIMIZER_MOMENTUM


def get_alxs(
        loss_function: Callable,
        learning_rate: float,
        number_of_classes: int
) -> Model:
    model = Sequential()
    model.add(Conv2D(
        input_shape=(227, 227, 3),
        filters=96,
        kernel_size=(7, 7),
        strides=4,
        activation=relu
    ))
    model.add(MaxPool2D(
        pool_size=(3, 3),
        padding='same',
        strides=2
    ))
    model.add(LocalResponseNormalization())
    model.add(Conv2D(
        filters=256,
        kernel_size=(5, 5),
        padding='same',
        activation=relu
    ))
    model.add(MaxPool2D(
        pool_size=(3, 3),
        padding='same',
        strides=2
    ))
    model.add(LocalResponseNormalization())
    model.add(Conv2D(
        filters=384,
        kernel_size=(3, 3),
        padding='same',
        activation=relu
    ))
    model.add(MaxPool2D(
        pool_size=(3, 3),
        padding='same',
        strides=2
    ))
    model.add(Flatten())
    model.add(Dense(
        units=512,
        activation=relu
    ))
    model.add(Dropout(0.5))
    model.add(Dense(
        units=512,
        activation=relu
    ))
    model.add(Dropout(0.5))
    model.add(Dense(
        units=number_of_classes,
        activation=softmax
    ))
    model.compile(
        loss=loss_function,
        optimizer=SGD(
            learning_rate=learning_rate,
            momentum=OPTIMIZER_MOMENTUM
        ),
        metrics=['acc']
    )
    return model


class LocalResponseNormalization(Layer):

    def __init__(self, n=5, alpha=0.0001, beta=0.75, k=2, **kwargs):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.k = k
        super(LocalResponseNormalization, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        _, f, r, c = inputs.shape
        squared = K.square(inputs)
        pooled = K.pool2d(
            squared,
            (self.n, self.n),
            strides=(1, 1),
            padding="same",
            pool_mode="avg"
        )
        summed = K.sum(pooled, axis=1, keepdims=True)
        averaged = self.alpha * K.repeat_elements(summed, f, axis=1)
        denominator = K.pow(self.k + averaged, self.beta)
        return inputs / denominator

    def compute_output_shape(self, input_shape):
        return input_shape
