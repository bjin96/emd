from typing import Callable

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD

from models.constants import OPTIMIZER_MOMENTUM


def get_model(
        loss_function: Callable,
        learning_rate: float,
        number_of_classes: int
) -> Model:
    vgg = VGG16(
        include_top=False,
        input_shape=(400, 400, 3),
        pooling='avg'
    )
    model = Sequential()
    model.add(vgg)
    model.add(Flatten())
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
