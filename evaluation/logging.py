from datetime import datetime
from pathlib import Path

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from data_handlers.data_set_info import DatasetName

LOG_DIR = Path('./logs')
CHECKPOINT_DIR = Path('./checkpoints')


def get_tensorboard_callback(
        data_set_name: DatasetName,
        learning_rate: float
) -> TensorBoard:
    current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    data_set_log_dir = LOG_DIR / Path(data_set_name.value) / Path(str(learning_rate)) / Path(current_datetime)
    data_set_log_dir.mkdir(parents=True, exist_ok=True)
    return TensorBoard(
        log_dir=data_set_log_dir
    )


def get_checkpoint_file(
        data_set_name: DatasetName,
        learning_rate: float
) -> ModelCheckpoint:
    current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    data_set_checkpoint_dir = CHECKPOINT_DIR / Path(data_set_name.value) / Path(str(learning_rate))
    data_set_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return ModelCheckpoint(
        filepath=str(data_set_checkpoint_dir / Path(current_datetime)),
        monitor='val_categorical_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
