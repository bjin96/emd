from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np

from data_handlers.generators import Y_COLUMN, X_COLUMN

FOLD_DIR = Path('./datasets/adience/')
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
IMAGES_DIR = './datasets/adience/faces/'
IMAGES_PREFIX = 'coarse_tilt_aligned_face.'

ADIENCE_CLASSES = ["(0, 2)", "(4, 6)", "(8, 12)", "(15, 20)", "(25, 32)", "(38, 43)", "(48, 53)", "(60, 100)"]


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
    info[X_COLUMN] = build_image_path_series(info=info)
    info[Y_COLUMN] = fix_outlier_labels(labels=info['age'])
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
        data=IMAGES_DIR + info['user_id'] + '/' + IMAGES_PREFIX + info['face_id'].astype(str)
        + '.' + info['original_image'],
        name=X_COLUMN
    )


def fix_outlier_labels(
        labels: pd.Series
) -> pd.Series:
    outlier_class_mapping = {
        '35': ADIENCE_CLASSES[5], '3': ADIENCE_CLASSES[0], '55': ADIENCE_CLASSES[7], '58': ADIENCE_CLASSES[7],
        '22': ADIENCE_CLASSES[3], '13': ADIENCE_CLASSES[2], '45': ADIENCE_CLASSES[5], '36': ADIENCE_CLASSES[5],
        '23': ADIENCE_CLASSES[4], '57': ADIENCE_CLASSES[7], '56': ADIENCE_CLASSES[6], '2': ADIENCE_CLASSES[0],
        '29': ADIENCE_CLASSES[4], '34': ADIENCE_CLASSES[4], '42': ADIENCE_CLASSES[5], '46': ADIENCE_CLASSES[6],
        '32': ADIENCE_CLASSES[4], '(38, 48)': ADIENCE_CLASSES[5], '(38, 42)': ADIENCE_CLASSES[5],
        '(8, 23)': ADIENCE_CLASSES[2], '(27, 32)': ADIENCE_CLASSES[4], 'None': np.nan
    }
    return labels.replace(
        to_replace=outlier_class_mapping
    )
