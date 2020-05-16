from typing import Callable

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Layer
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.optimizers import SGD
from tensorflow.python.ops.nn_ops import local_response_normalization

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
        return local_response_normalization(
            input=inputs,
            depth_radius=self.n,
            bias=self.k,
            alpha=self.alpha,
            beta=self.beta
        )

    def compute_output_shape(self, input_shape):
        return input_shape
