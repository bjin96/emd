from typing import Callable

import tensorflow.keras.backend as K

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Conv2D, Layer, ReLU, BatchNormalization, Dropout, Dense
from tensorflow.keras.models import Model, Sequential

from models.constants import OPTIMIZER_MOMENTUM


def get_res(
        loss_function: Callable,
        learning_rate: float,
        number_of_classes: int
) -> Model:
    wrn = WideResidualNetwork(
        group_size=13,
        activation=ReLU,
    )
    model = Sequential()
    model.add(wrn)
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(number_of_classes, activation='softmax'))
    model.compile(
        loss=loss_function,
        optimizer=SGD(
            momentum=OPTIMIZER_MOMENTUM,
            learning_rate=learning_rate
        ),
        metrics=['acc']
    )
    return model


def get_res_f(
        loss_function: Callable,
        learning_rate: float,
        number_of_classes: int
) -> Model:
    pass


class BasicLayer(Layer):

    def __init__(self, **kwargs):
        super(BasicLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = self.sub_layers[0](inputs)
        for layer in self.sub_layers[1:]:
            x = layer(x)
        return x


class ConvolutionBlock(BasicLayer):

    def __init__(self, filters, kernel_size, stride=1, activation=ReLU, k=1, **kwargs):
        super(ConvolutionBlock, self).__init__(**kwargs)
        self.sub_layers = [
            BatchNormalization(),
            activation(),
            Conv2D(
                filters=filters * k,
                kernel_size=kernel_size,
                strides=stride,
                padding='same'
            )
        ]

    def call(self, inputs, **kwargs):
        super(ConvolutionBlock, self).call(inputs, **kwargs)


class BottleneckBlock(BasicLayer):

    def __init__(self, filters, stride=1, activation=relu, k=1, **kwargs):
        super(BottleneckBlock, self).__init__(**kwargs)
        self.sub_layers = [
            ConvolutionBlock(
                filters=filters * k,
                kernel_size=(1, 1),
                stride=stride,
                activation=activation
            ),
            ConvolutionBlock(
                filters=filters * k,
                kernel_size=(3, 3),
                activation=activation
            ),
            ConvolutionBlock(
                filters=filters * k,
                kernel_size=(1, 1),
                activation=activation
            )
        ]

    def call(self, inputs, **kwargs):
        super(BottleneckBlock, self).call(inputs, **kwargs)


class Group(BasicLayer):

    def __init__(self, n, filters, stride=1, activation=ReLU, k=1, **kwargs):
        super(Group, self).__init__(**kwargs)
        self.sub_layers = BottleneckBlock(filters, stride, activation, k)
        self.sub_layers.extend([BottleneckBlock(filters, activation=activation, k=k) for _ in range(n - 1)])

    def call(self, inputs, **kwargs):
        super(Group, self).call(inputs, **kwargs)


class WideResidualNetwork(Model):

    FILTER_SIZES = [16, 32, 64]
    STRIDES = [1, 2, 2]

    def __init__(self, group_size, activation=ReLU, k=1, **kwargs):
        super(WideResidualNetwork, self).__init__(**kwargs)
        self.groups = [
            Conv2D(
                filters=WideResidualNetwork.FILTER_SIZES[0],
                kernel_size=(3, 3),
                activation=activation
            )
        ]
        self.groups.extend([
            Group(
                n=group_size,
                filters=WideResidualNetwork.FILTER_SIZES[i],
                stride=WideResidualNetwork.STRIDES[i],
                activation=activation,
                k=k
            )
            for i in range(len(WideResidualNetwork.FILTER_SIZES))
        ])

    def call(self, inputs, **kwargs):
        x = self.groups[0](inputs)
        for group in self.groups[1:]:
            x = group(x)
        return x
