from pathlib import Path
from typing import List, Callable, Union, Type

from tensorflow.keras.activations import linear, softmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard

from data_handlers.adience import get_adience_info, ADIENCE_TRAIN_FOLDS_INFO_FILES, \
    ADIENCE_VALIDATION_FOLDS_INFO_FILES, ADIENCE_CLASSES
from data_handlers.data_set_info import DatasetName
from data_handlers.generators import get_generators
from evaluation.logging import get_checkpoint_file, get_tensorboard_callback
from models.constants import LEARNING_RATES
from models.evaluation_model import EvaluationModel


def evaluate_adience_model(
        evaluation_model: Type[EvaluationModel],
        loss_function: Callable,
        final_activation: Union[softmax, linear],
        **loss_function_kwargs
):
    for learning_rate in LEARNING_RATES:
        model = evaluation_model(
            number_of_classes=len(ADIENCE_CLASSES),
            final_activation=final_activation,
            loss_function=loss_function,
            learning_rate=learning_rate,
            **loss_function_kwargs
        )
        evaluate_adience_folds(
            model=model,
            checkpoint_callback=get_checkpoint_file(
                data_set_name=DatasetName.ADIENCE,
                learning_rate=learning_rate
            ),
            tensorboard_callback=get_tensorboard_callback(
                data_set_name=DatasetName.ADIENCE,
                learning_rate=learning_rate
            )
        )


def evaluate_adience_folds(
        model: EvaluationModel,
        checkpoint_callback: ModelCheckpoint,
        tensorboard_callback: TensorBoard
) -> None:
    evaluate_adience_fold(
        model=model,
        train_fold_info_files=ADIENCE_TRAIN_FOLDS_INFO_FILES[0],
        validation_fold_info_file=ADIENCE_VALIDATION_FOLDS_INFO_FILES[0],
        checkpoint_callback=checkpoint_callback,
        tensorboard_callback=tensorboard_callback
    )


def evaluate_adience_fold(
        model: EvaluationModel,
        train_fold_info_files: List[Path],
        validation_fold_info_file: Path,
        checkpoint_callback: ModelCheckpoint,
        tensorboard_callback: TensorBoard
) -> None:
    train_info, validation_info = get_adience_info(
        train_fold_info_files=train_fold_info_files,
        validation_fold_info_file=validation_fold_info_file
    )
    train_generator, validation_generator = get_generators(
        train_info=train_info,
        validation_info=validation_info
    )
    evaluate(
        model=model,
        train_generator=train_generator,
        validation_generator=validation_generator,
        checkpoint_callback=checkpoint_callback,
        tensorboard_callback=tensorboard_callback
    )


def evaluate(
        model: EvaluationModel,
        train_generator: ImageDataGenerator,
        validation_generator: ImageDataGenerator,
        checkpoint_callback: ModelCheckpoint,
        tensorboard_callback: TensorBoard
) -> None:
    model.fit(
        x=train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[checkpoint_callback, tensorboard_callback]
    )
