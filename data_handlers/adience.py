from pathlib import Path
from typing import List, Tuple

import pandas as pd

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

ADIENCE_NUMBER_OF_CLASSES = 8


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
    info['x_col'] = IMAGES_DIR + info['user_id'] + '/' + IMAGES_PREFIX + info['original_image']
    info['y_col'] = info['age']
    return info.drop(
        labels=['user_id', 'original_image', 'face_id', 'gender', 'age', 'x', 'y', 'dx', 'dy', 'tilt_ang',
                'fiducial_yaw_angle', 'fiducial_score'],
        axis='columns'
    )
