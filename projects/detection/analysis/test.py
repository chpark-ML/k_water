import logging
import os
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import pandas as pd
import tqdm
from omegaconf import DictConfig, OmegaConf

import projects.common.constants as C
import projects.detection.model_path as MP
from projects.detection.datasets.k_water import collater
from projects.detection.criterions.coco_eval import evaluate_coco
from projects.detection.train import _get_loaders_and_trainer
from projects.common.enums import RunMode
from projects.detection.utils import set_config
from projects.detection.utils import (_seed_everything, print_config, set_config, get_torch_device_string)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_checkpoint(model, ckpt_path, device):
    """Loads checkpoint from directory"""
    assert os.path.exists(ckpt_path)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=True)
    logger.info(f'Model loaded from {ckpt_path}')

    return model


def main() -> None:
    # checkpoint path
    ckpt_path = MP.TARGET_MODEL_PATH / Path("model.pth")
    config_path = MP.TARGET_MODEL_PATH / Path(".hydra/config.yaml")
    config = OmegaConf.load(config_path)
    
    if config.model.get('target_function') is not None:
        logger.info(f'Instantiating model <{config.model.target_function}>')
        target_function = hydra.utils.get_method(config.model.target_function)
        config.model.pop('target_function')
        config.model["pretrained"] = False
        model = target_function(**config.model).cuda()
    else:
        logger.info(f'Instantiating model <{config.model._target_}>')
        model = hydra.utils.instantiate(config.model)
    
    gpus = 0
    torch_device = f'cuda:{gpus}'
    # torch_device = f'cpu'

    if torch_device.startswith('cuda'):
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    device = torch.device(torch_device)
    model = load_checkpoint(model, ckpt_path, device)

    run_modes = [RunMode('test')]
    datasets = {mode: hydra.utils.instantiate(config.loader.dataset, mode=mode) 
                for mode in run_modes}
    loaders = {mode: hydra.utils.instantiate(config.loader,
                                             dataset=datasets[mode], 
                                             collate_fn=collater,
                                             shuffle=(mode == RunMode.TRAIN),
                                             drop_last=(mode == RunMode.TRAIN)) for mode in run_modes}
    evaluate_coco(loaders[RunMode.TEST].dataset, model, threshold=0.05, device=torch_device)

if __name__ == '__main__':
    main()
