import os
from pathlib import Path

from pycocotools.coco import COCO

DATA_ROOT_PATH = Path('/data/k_water/dataset')
DATA_ROOT_PATH_TRAIN = DATA_ROOT_PATH / 'train'
DATA_ROOT_PATH_TEST = DATA_ROOT_PATH / 'test'
NUM_FOLD = 10

TRAIN_DATA_IMAGE = sorted(DATA_ROOT_PATH_TRAIN.rglob("train_*.png"))
TEST_DATA_IMAGE = sorted(DATA_ROOT_PATH_TEST.rglob("test_*.png"))
TRAIN_DATA_ANNOT = DATA_ROOT_PATH / "labels" / "train.json"

ANSWER_SAMPLE = DATA_ROOT_PATH / "labels" / "answer_sample.json"

coco=COCO(TRAIN_DATA_ANNOT)
NUM_CLASS = len(coco.getCatIds())
IMAGE_FEATURES = ['img_id', "num_annot", "average_area", "entropy"]
for i in range(NUM_CLASS):
    IMAGE_FEATURES.append(f"is_class_{i}")
