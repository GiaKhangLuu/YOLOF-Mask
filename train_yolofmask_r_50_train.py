import sys
sys.path.insert(0, './detectron2/detectron2')

import argparse
import sys
import os
import torch, detectron2

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.model_zoo import get_config
from detectron2.config import LazyConfig
from detectron2.config.instantiate import instantiate
from detectron2.engine import default_setup

from tools.lazyconfig_train_net import do_train

dataset = 'bdd100k'
annot_dir = f"./{dataset}/labels_coco/ins_seg"
imgs_dir = "./{}/images/10k/{}"

for split in ['train', 'val']:
    annot_path = os.path.join(annot_dir, f"ins_seg_{split}_coco.json")
    d_name = dataset + f'_{split}'
    register_coco_instances(d_name, {}, annot_path, imgs_dir.format(dataset, split))

# Load dataset
dataset_dicts = DatasetCatalog.get('bdd100k_train')
metadata = MetadataCatalog.get('bdd100k_train')

config_file = "yolof_mask/configs/yolof_mask/yolofmask_r_50_1x.py"

class Args(argparse.Namespace):
    config_file=config_file
    eval_only=False
    num_gpus=1
    num_machines=1
    resume=False

args = Args()

cfg = LazyConfig.load(config_file)

default_setup(cfg, args)

do_train(args, cfg)