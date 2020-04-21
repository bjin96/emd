from typing import Callable

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD

OPTIMIZER_MOMENTUM = 0.98


def get_model(
        loss_function: Callable,
        learning_rate: float,
        number_of_classes: int
) -> Model:
    vgg = VGG16(
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    model = Sequential()
    model.add(vgg)
    model.add(Flatten())
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
