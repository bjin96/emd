from pathlib import Path
from typing import List, Callable, Union, Type

from tensorflow.keras.activations import linear, softmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_handlers.adience import get_adience_info, ADIENCE_TRAIN_FOLDS_INFO_FILES, \
    ADIENCE_VALIDATION_FOLDS_INFO_FILES, ADIENCE_CLASSES
from data_handlers.data_set_info import DatasetName
from data_handlers.generators import custom_data_loading
from models.constants import LEARNING_RATES
from models.evaluation_model import EvaluationModel


def evaluate_adience_model(
        evaluation_model: Type[EvaluationModel],
        learning_rate_index: int,
        fold_index: int,
        loss_function: Callable,
        final_activation: Union[softmax, linear],
        ground_distance_path: Path = None,
        **loss_function_kwargs
):
    """
    Function to evaluate a model with the specified parameters on the Adience dataset.

    :param evaluation_model: Model to be evaluated.
    :param learning_rate_index: Index of the learning rate with which to train. Learning rates are defined in
        emd/models/constants.py.
    :param fold_index: Index of the fold from the cross-validation partitioning on which to train the model.
    :param loss_function: Loss function with which to train the model.
    :param final_activation: Final activation to use for the final layer. Must be one of softmax or linear.
    :param ground_distance_path: Path where the ground distance matrices should be stored if the loss function is
        self-guided emd.
    :param loss_function_kwargs: Additional parameters to pass to the loss function.
    """
    model = evaluation_model(
        number_of_classes=len(ADIENCE_CLASSES),
        dataset_name=DatasetName.ADIENCE,
        final_activation=final_activation,
        loss_function=loss_function,
        learning_rate=LEARNING_RATES[learning_rate_index],
        fold_index=fold_index,
        ground_distance_path=ground_distance_path,
        **loss_function_kwargs
    )
    _evaluate_adience_fold(
        model=model,
        train_fold_info_files=ADIENCE_TRAIN_FOLDS_INFO_FILES[fold_index],
        validation_fold_info_file=ADIENCE_VALIDATION_FOLDS_INFO_FILES[fold_index]
    )


def _evaluate_adience_fold(
        model: EvaluationModel,
        train_fold_info_files: List[Path],
        validation_fold_info_file: Path
) -> None:
    train_info, validation_info = get_adience_info(
        train_fold_info_files=train_fold_info_files,
        validation_fold_info_file=validation_fold_info_file
    )
    train_generator, validation_generator = custom_data_loading(
        train_info=train_info,
        validation_info=validation_info
    )
    _evaluate(
        model=model,
        train_generator=train_generator,
        validation_generator=validation_generator
    )


def _evaluate(
        model: EvaluationModel,
        train_generator: ImageDataGenerator,
        validation_generator: ImageDataGenerator
) -> None:
    model.train(
        x=train_generator,
        epochs=160,
        validation_data=validation_generator,
        steps_per_epoch=None,
        validation_steps=None
    )
