import os
from datetime import datetime
from pathlib import Path

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from data_handlers.data_set_info import DatasetName

LOG_DIR = os.path.dirname(__file__) / Path('../logs')
CHECKPOINT_DIR = os.path.dirname(__file__) / Path('../checkpoints')


def get_tensorboard_callback(
        loss_name: str,
        data_set_name: DatasetName,
        learning_rate: float,
        model_name: str,
        fold_index: int
) -> TensorBoard:
    """Function for adding logging to TensorBoard."""
    current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    data_set_log_dir = \
        LOG_DIR / Path(data_set_name.value) / Path(model_name) / Path(loss_name) / Path(str(fold_index)) \
        / Path(str(learning_rate)) / Path(current_datetime)
    data_set_log_dir.mkdir(parents=True, exist_ok=True)
    return TensorBoard(
        log_dir=data_set_log_dir
    )


def get_checkpoint_file(
        loss_name: str,
        data_set_name: DatasetName,
        learning_rate: float,
        model_name: str,
        fold_index: int
) -> ModelCheckpoint:
    """Function for generating checkpoints during training."""
    current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    data_set_checkpoint_dir = \
        CHECKPOINT_DIR / Path(data_set_name.value) / Path(model_name) / Path(loss_name) \
        / Path(str(fold_index)) / Path(str(learning_rate))
    data_set_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return ModelCheckpoint(
        filepath=str(data_set_checkpoint_dir / Path(current_datetime)),
        monitor='val_categorical_accuracy'
    )
