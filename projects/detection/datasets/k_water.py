import os
import random
from typing import List, Dict, Optional, Sequence, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
from omegaconf import OmegaConf
import skimage.io
import skimage.transform
import skimage.color
import skimage

from projects.common.enums import RunMode
import projects.common.constants as C

_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_CSV_FILE_PATH = os.path.join(_THIS_DIR, '../..', 'database', 'image_info.csv')


def _get_df(mode: RunMode, dataset_info: dict):
    dataset_size_scale_factor = dataset_info['dataset_size_scale_factor']
    assert (dataset_size_scale_factor > 0.) and (dataset_size_scale_factor <= 1.)
    
    # get dataframe for each dataset
    df = pd.read_csv(_CSV_FILE_PATH)

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
        # TODO: shuffle
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

        if self.mode == RunMode.TRAIN or self.mode == RunMode.VALIDATE:
            self.df_data = _get_df(self.mode, self.dataset_info)
            self.coco = COCO(C.TRAIN_DATA_ANNOT)
            self.load_classes()
        elif self.mode == RunMode.TEST:
            self.df_data = pd.DataFrame({
                'img_id': [int(image_path.stem.split('_')[-1]) for image_path in C.TEST_DATA_IMAGE], 
                'img_path': C.TEST_DATA_IMAGE,
            })


        if self.mode == RunMode.TRAIN:
            self.transforms = Compose(transforms=[Normalizer(), Augmenter(), Resizer()])
        else:
            self.transforms = Compose(transforms=[Normalizer(), Resizer()])

    
    def __len__(self):
        return len(self.df_data)
    
    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, int]]:
        """
        Resize -> Windowing -> Additional data augmentation
        """
        elem = self.df_data.iloc[index]
        image_id = elem['img_id']

        if self.mode == RunMode.TEST:
            images = self.load_image(image_id, elem['img_path'])
            annotations = np.zeros((0, 5))
        else:
            images = self.load_image(image_id)
            annotations = self.load_annotations(image_id)
        sample = {'img': images, 'annot': annotations}

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def load_image(self, image_id, image_path=None):
        if image_path:
            path = image_path
        else:
            image_info = self.coco.loadImgs([image_id])[0]
            path       = os.path.join(os.path.join(str(C.DATA_ROOT_PATH_TRAIN), image_info['file_name']))
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0
    
    def load_annotations(self, image_id):

        # coco
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        coco_annotations = self.coco.loadAnns(ann_ids)
        num_objs = len(coco_annotations)
        annotations = np.zeros((0, 5))
        if len(ann_ids) == 0:
            return annotations
        
        # get annot
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation        = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]


    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def num_classes(self):
        return C.NUM_CLASS
    
def collater(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)
    
    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=480, max_side=1024):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side
        # scale = 0.9 + (1.1 - 0.9) * np.random.rand()

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32 if rows%32 else 0
        pad_h = 32 - cols%32 if cols%32 else 0

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        if annots is not None:
            annots[:, :4] *= scale
            return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}
        else:
            return {'img': torch.from_numpy(new_image), 'annot': None, 'scale': scale}
        


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            
            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']

        return {'img':((image.astype(np.float32)-self.mean)/self.std), 'annot': annots}

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
