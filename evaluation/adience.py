from pathlib import Path
from typing import List, Callable

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard

from data_handlers.adience import get_adience_info, ADIENCE_TRAIN_FOLDS_INFO_FILES, \
    ADIENCE_VALIDATION_FOLDS_INFO_FILES, ADIENCE_CLASSES
from data_handlers.data_set_info import DatasetName
from data_handlers.generators import get_generators
from evaluation.logging import get_checkpoint_file, get_tensorboard_callback
from models.alx import get_alxs
from models.constants import LEARNING_RATES
from models.vgg import get_model


def evaluate_adience_vgg(
        loss_function: Callable
) -> None:
    for learning_rate in LEARNING_RATES:
        model = get_model(
            loss_function=loss_function,
            learning_rate=learning_rate,
            number_of_classes=len(ADIENCE_CLASSES)
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


def evaluate_adience_alxs(
        loss_function: Callable
) -> None:
    for learning_rate in LEARNING_RATES:
        model = get_alxs(
            loss_function=loss_function,
            learning_rate=learning_rate,
            number_of_classes=len(ADIENCE_CLASSES)
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
        model: Model,
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
        model: Model,
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
        model: Model,
        train_generator: ImageDataGenerator,
        validation_generator: ImageDataGenerator,
        checkpoint_callback: ModelCheckpoint,
        tensorboard_callback: TensorBoard
) -> None:
    model.fit(
        train_generator,
        epochs=1,
        validation_data=validation_generator,
        callbacks=[checkpoint_callback, tensorboard_callback]
    )
