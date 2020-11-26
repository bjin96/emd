from abc import ABC, abstractmethod
from typing import ClassVar, Callable, Union, List

from tensorflow.keras.metrics import Metric
from tensorflow.keras import Model
from tensorflow.keras.activations import linear, softmax
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import SGD, Optimizer
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow import TensorShape

from data_handlers.data_set_info import DatasetName
from evaluation.logging import get_checkpoint_file, get_tensorboard_callback
from loss_functions.emd import EmdWeightHeadStart
from metrics.accuracy import one_off_accuracy


class EvaluationModel(ABC, Model):

    _OPTIMIZER: ClassVar[Optimizer] = SGD
    _OPTIMIZER_MOMENTUM: ClassVar[float] = 0.98
    _METRICS: ClassVar[List[Metric]] = [
        categorical_accuracy,
        one_off_accuracy
    ]
    _MODEL_NAME: ClassVar[str] = 'base_model'
    model: Model = None

    def __init__(
            self,
            number_of_classes: int,
            dataset_name: DatasetName,
            final_activation: Union[softmax, linear],
            loss_function: Callable,
            learning_rate: float,
            fold_index: int,
            **loss_function_kwargs,
    ):
        super(EvaluationModel, self).__init__()
        self.number_of_classes = number_of_classes
        self.learning_rate = learning_rate
        self.fold_index = fold_index
        self.dataset_name = dataset_name
        self.loss_function = loss_function
        self.second_to_last_layer = None

        self._build_model(
            number_of_classes=number_of_classes,
            final_activation=final_activation
        )
        self._compile_model(
            loss_function=loss_function,
            **loss_function_kwargs
        )

    def compute_output_shape(self, input_shape):
        return TensorShape((
            input_shape[0],
            self.number_of_classes
        ))

    @abstractmethod
    def _build_model(
            self,
            number_of_classes: int,
            final_activation: Union[softmax, linear]
    ):
        pass

    def call(self, inputs, **kwargs):
        y = inputs
        for layer in self.layers[:-2]:
            y = layer(y, **kwargs)
        self.second_to_last_layer = self.layers[-2](y, **kwargs)
        output = self.layers[-1](self.second_to_last_layer, **kwargs)
        if not kwargs['training']:
            y = inputs[:, :, ::-1, :]
            for layer in self.layers[:-2]:
                y = layer(y, **kwargs)
            self.second_to_last_layer = self.layers[-2](y, **kwargs)
            mirrored_output = self.layers[-1](self.second_to_last_layer, **kwargs)
            return (mirrored_output + output) / 2
        else:
            return output

    def _compile_model(
            self,
            loss_function: Callable,
            **loss_function_kwargs
    ):
        self.emd_weight_head_start = EmdWeightHeadStart()
        lr_schedule = ExponentialDecay(
            self.learning_rate,
            decay_steps=429,
            decay_rate=0.995
        )
        self.compile(
            loss=loss_function(
                model=self,
                **loss_function_kwargs
            ),
            optimizer=self._OPTIMIZER(
                learning_rate=lr_schedule,
                nesterov=True,
                momentum=self._OPTIMIZER_MOMENTUM
            ),
            metrics=self._METRICS,
            run_eagerly=False
        )

    def test(self, **kwargs):
        return self.predict(**kwargs)

    def train(self, **kwargs):
        return self.fit(
            callbacks=[
                self.emd_weight_head_start,
                get_checkpoint_file(
                    loss_name=self.loss_function.__name__,
                    data_set_name=self.dataset_name,
                    learning_rate=self.learning_rate,
                    model_name=self._MODEL_NAME,
                    fold_index=self.fold_index
                ),
                get_tensorboard_callback(
                    loss_name=self.loss_function.__name__,
                    data_set_name=self.dataset_name,
                    learning_rate=self.learning_rate,
                    model_name=self._MODEL_NAME,
                    fold_index=self.fold_index
                )
            ],
            **kwargs
        )
