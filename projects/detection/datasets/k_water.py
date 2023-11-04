import os
import random
from typing import List, Dict, Optional, Sequence, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
from omegaconf import OmegaConf

from projects.common.enums import RunMode
import projects.common.constants as C
import projects.detection.utils.augmentation as aug


def _get_df(mode: RunMode, dataset_info: dict):
    dataset_size_scale_factor = dataset_info['dataset_size_scale_factor']
    assert (dataset_size_scale_factor > 0.) and (dataset_size_scale_factor <= 1.)
    
    # get dataframe for each dataset
    df = pd.read_csv("projects/database/image_info.csv")

    # get fold indices depending on the "mode"
    if mode == RunMode.TRAIN:
        total_fold = dataset_info["total_fold"]
        val_fold = dataset_info["val_fold"]
        test_fold = dataset_info["test_fold"]
        fold_indices_total = OmegaConf.to_container(total_fold, resolve=True)
        fold_indices_val = OmegaConf.to_container(val_fold, resolve=True)
        fold_indices_test = OmegaConf.to_container(test_fold, resolve=True)
        fold_indices = [item for item in fold_indices_total if
                        item not in fold_indices_val and item not in fold_indices_test]
    elif mode == RunMode.VALIDATE:
        val_fold = dataset_info["val_fold"]
        fold_indices = OmegaConf.to_container(val_fold, resolve=True)
    elif mode == RunMode.TEST:
        test_fold = dataset_info["test_fold"]
        fold_indices = OmegaConf.to_container(test_fold, resolve=True)
    else:
        assert False, "fold selection did not work as intended."

    # get specific size of samples from each fold index
    dfs = list()
    for i_fold in fold_indices:
        _df = df[df['fold'] == i_fold].copy()

        # scale dataset size for train dataset
        _df = _df[:int(len(_df) * dataset_size_scale_factor)]

        # append dataframes
        dfs.append(_df)

    df = pd.concat(dfs, ignore_index=True)

    return df


class Compose:
    def __init__(self, transforms=None, mode: Union[str, RunMode] = 'train'):
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.transforms = transforms

    def __call__(self, x):
        for f in self.transforms:
            x= f(x)

        return x
        

class KWATER(Dataset):
    def __init__(self,
                 mode: Union[str, RunMode],
                 dataset_info: dict = None,
                 augmentation: dict = None):
        self.mode: RunMode = RunMode(mode) if isinstance(mode, str) else mode
        self.dataset_info = dataset_info
        self.df_data = _get_df(self.mode, self.dataset_info)
        self.coco = COCO(C.TRAIN_DATA_ANNOT)
        self.cat_ids = self.coco.getCatIds() # category id 반환
        self.cats = self.coco.loadCats(self.cat_ids) # category id를 입력으로 category name, super category 정보 담긴 dict 반환

        self.transforms = Compose(transforms=[
            transforms.ToPILImage(),
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
            ])
    
    def __len__(self):
        return len(self.df_data)
    
    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, int]]:
        """
        Resize -> Windowing -> Additional data augmentation
        """
        elem = self.df_data.iloc[index]
        image_id = elem['img_id']
        image_infos = self.coco.loadImgs(image_id)[0] # img id를 받아서 image info 반환
        
        # cv2 를 활용하여 image 불러오기(BGR -> RGB 변환 -> numpy array 변환 -> normalize(0~1))
        images = cv2.imread(os.path.join(str(C.DATA_ROOT_PATH_TRAIN), image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0 

        # bounding boxes
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        num_objs = len(annotations)
        boxes = []
        for i in range(num_objs):
            xmin = annotations[i]['bbox'][0]
            ymin = annotations[i]['bbox'][1]
            xmax = xmin + annotations[i]['bbox'][2]
            ymax = ymin + annotations[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([image_id])
        areas = []
        for i in range(num_objs):
            areas.append(annotations[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        
        # target in dictionary format
        target = {}
        target["image_id"] = image_id
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = areas

        if self.transforms is not None:
            images, target = self.transforms(images, target)

        return images, target