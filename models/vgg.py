from typing import Union, ClassVar

from tensorflow.keras import Sequential
from tensorflow.keras.activations import softmax, linear, relu
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications.vgg16 import VGG16

from models.evaluation_model import EvaluationModel


class Vggf(EvaluationModel):

    _MODEL_NAME: ClassVar[str] = 'VGG_f'

    def _build_model(
            self,
            number_of_classes: int,
            final_activation: Union[softmax, linear]
    ):
        vgg = VGG16(
            include_top=False,
            input_shape=(227, 227, 3),
            pooling='avg'
        )
        self.model = Sequential()
        self.model.add(vgg)
        self.model.add(Dense(128, activation=relu))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(128, activation=relu))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(number_of_classes, activation=final_activation))
