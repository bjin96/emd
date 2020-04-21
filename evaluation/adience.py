from pathlib import Path
from typing import List

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

from data_handlers.adience import get_adience_info, ADIENCE_NUMBER_OF_CLASSES, ADIENCE_TRAIN_FOLDS_INFO_FILES, \
    ADIENCE_VALIDATION_FOLDS_INFO_FILES
from data_handlers.generators import get_generators
from loss_functions.emd import earth_mover_distance
from models.vgg import get_model

LEARNING_RATES = [10 ** -1.5, 10 ** -2, 10 ** -2.5, 10 ** -3, 10 ** -3.5, 10 ** -4, 10 ** -4.5]
CHECKPOINTS_DIR = Path('./checkpoints/')
ADIENCE_CHECKPOINT_FILE = CHECKPOINTS_DIR / Path('adience_checkpoint.h5')


def evaluate_adience_vgg():
    model = get_model(
        loss_function=earth_mover_distance,
        learning_rate=LEARNING_RATES[0],
        number_of_classes=ADIENCE_NUMBER_OF_CLASSES
    )
    evaluate_adience_folds(
        model=model
    )


def evaluate_adience_folds(
        model: Model
) -> None:
    for i in range(len(ADIENCE_TRAIN_FOLDS_INFO_FILES)):
        evaluate_adience_fold(
            model=model,
            train_fold_info_files=ADIENCE_TRAIN_FOLDS_INFO_FILES[i],
            validation_fold_info_file=ADIENCE_VALIDATION_FOLDS_INFO_FILES[i]
        )


def evaluate_adience_fold(
        model: Model,
        train_fold_info_files: List[Path],
        validation_fold_info_file: Path
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
        checkpoint_file=ADIENCE_CHECKPOINT_FILE
    )


def evaluate(
        model: Model,
        train_generator: ImageDataGenerator,
        validation_generator: ImageDataGenerator,
        checkpoint_file: Path
) -> None:
    model.fit_generator(
        train_generator,
        steps_per_epoch=350,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=35,
        callbacks=[
            ModelCheckpoint(
                filepath=checkpoint_file,
                monitor='val_acc',
                verbose=1,
                save_best_only=True,
                mode='max'
            )
        ]
    )
