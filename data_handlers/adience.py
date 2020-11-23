import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np


FOLD_DIR = os.path.dirname(__file__) / Path('../datasets/adience/')
FOLD_0_FILE = FOLD_DIR / Path('fold_0_data.csv')
FOLD_1_FILE = FOLD_DIR / Path('fold_1_data.csv')
FOLD_2_FILE = FOLD_DIR / Path('fold_2_data.csv')
FOLD_3_FILE = FOLD_DIR / Path('fold_3_data.csv')
FOLD_4_FILE = FOLD_DIR / Path('fold_4_data.csv')
TRAIN_0_INFO_FILES = [FOLD_0_FILE, FOLD_1_FILE, FOLD_2_FILE, FOLD_3_FILE]
TRAIN_1_INFO_FILES = [FOLD_0_FILE, FOLD_1_FILE, FOLD_2_FILE, FOLD_4_FILE]
TRAIN_2_INFO_FILES = [FOLD_0_FILE, FOLD_1_FILE, FOLD_3_FILE, FOLD_4_FILE]
TRAIN_3_INFO_FILES = [FOLD_0_FILE, FOLD_2_FILE, FOLD_3_FILE, FOLD_4_FILE]
TRAIN_4_INFO_FILES = [FOLD_1_FILE, FOLD_2_FILE, FOLD_3_FILE, FOLD_4_FILE]
VALIDATION_0_INFO_FILE = FOLD_4_FILE
VALIDATION_1_INFO_FILE = FOLD_3_FILE
VALIDATION_2_INFO_FILE = FOLD_2_FILE
VALIDATION_3_INFO_FILE = FOLD_1_FILE
VALIDATION_4_INFO_FILE = FOLD_0_FILE
ADIENCE_TRAIN_FOLDS_INFO_FILES = [TRAIN_0_INFO_FILES, TRAIN_1_INFO_FILES, TRAIN_2_INFO_FILES, TRAIN_3_INFO_FILES, TRAIN_4_INFO_FILES]
ADIENCE_VALIDATION_FOLDS_INFO_FILES = [
    VALIDATION_0_INFO_FILE, VALIDATION_1_INFO_FILE, VALIDATION_2_INFO_FILE, VALIDATION_3_INFO_FILE,
    VALIDATION_4_INFO_FILE
]
IMAGES_DIR = os.path.dirname(__file__) / Path('../datasets/adience/standardized/')
IMAGES_PREFIX = 'coarse_tilt_aligned_face.'

ADIENCE_CLASSES = ["(0, 2)", "(4, 6)", "(8, 12)", "(15, 20)", "(25, 32)", "(38, 43)", "(48, 53)", "(60, 100)"]
class_to_index = {
    '(0, 2)': 0,
    '(4, 6)': 1,
    '(8, 12)': 2,
    '(15, 20)': 3,
    '(25, 32)': 4,
    '(38, 43)': 5,
    '(48, 53)': 6,
    '(60, 100)': 7
}

def get_adience_info(
        train_fold_info_files: List[Path],
        validation_fold_info_file: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_info = []
    for train_info_file in train_fold_info_files:
        train_info.append(preprocess_info_file(train_info_file))
    validation_info = preprocess_info_file(validation_fold_info_file)
    return pd.concat(train_info), validation_info


def preprocess_info_file(
        info_file: Path
) -> pd.DataFrame:
    info = pd.read_csv(
        filepath_or_buffer=info_file,
        sep='\t'
    )
    info = info.loc[info['age'].isin(ADIENCE_CLASSES)]
    info['x_col'] = build_image_path_series(info=info)
    info['y_col'] = info['age'].map(lambda age: class_to_index[age])
    return info \
        .drop(
            labels=['user_id', 'original_image', 'face_id', 'gender', 'age', 'x', 'y', 'dx', 'dy', 'tilt_ang',
                    'fiducial_yaw_angle', 'fiducial_score'],
            axis='columns'
        ) \
        .dropna()


def build_image_path_series(
        info: pd.DataFrame
) -> pd.Series:
    return pd.Series(
        data=str(IMAGES_DIR) + '/' + info['user_id'] + '/' + IMAGES_PREFIX + info['face_id'].astype(str)
        + '.' + info['original_image'],
        name='x_col'
    )
