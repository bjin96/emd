from typing import Callable, Union

from tensorflow.keras.metrics import TopKCategoricalAccuracy, Accuracy
from tensorflow.keras.activations import softmax, linear, relu
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.vgg16 import VGG16

from models.constants import OPTIMIZER_MOMENTUM


def get_vgg_f(
        loss_function: Callable,
        learning_rate: float,
        number_of_classes: int,
        final_activation: Union[softmax, linear]
) -> Model:
    vgg = VGG16(
        include_top=False,
        input_shape=(227, 227, 3),
        pooling='avg'
    )
    model = Sequential()
    model.add(vgg)
    model.add(Dense(128, activation=relu))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation=relu))
    model.add(Dropout(0.2))
    model.add(Dense(number_of_classes, activation=final_activation))
    model.compile(
        loss=loss_function,
        optimizer=SGD(
            momentum=OPTIMIZER_MOMENTUM,
            learning_rate=learning_rate
        ),
        metrics=[Accuracy(), TopKCategoricalAccuracy(k=3)]
    )
    return model
