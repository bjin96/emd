from typing import Union, ClassVar

from tensorflow.keras.activations import softmax, linear, relu
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.vgg16 import VGG16

from models.evaluation_model import EvaluationModel


class Vggf(EvaluationModel):
    """
    VGG model. Reference: K. Simonyan and A. Zisserman. "Very deep convolutional networks for large-scale image
    recognition", 2014. Here, a pre-trained version of the network from tensorflow.keras.applications is used.
    """

    _MODEL_NAME: ClassVar[str] = 'VGG_f'

    def _build_model(
            self,
            number_of_classes: int,
            final_activation: Union[softmax, linear]
    ):
        self.vgg = VGG16(
            include_top=False,
            input_shape=(227, 227, 3),
            pooling='avg'
        )
        self.dense1 = Dense(4096, activation=relu)
        self.dense2 = Dense(4096, activation=relu)
        self.dense3 = Dense(number_of_classes, activation=final_activation)
