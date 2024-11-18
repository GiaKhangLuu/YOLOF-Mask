from ..common.train import train
from ..common.yolof_optim import YOLOF_SGD as optimizer
from ..common.yolof_coco_schedule import lr_multiplier_3x_b16 as lr_multiplier
from ..common.data.coco import dataloader
from ..common.models.yolof_mask_r_50 import model

default_batch_size = 16
batch_size = 16

dataloader.train.mapper.use_instance_mask = True
dataloader.train.mapper.instance_mask_format = "bitmask"
dataloader.train.total_batch_size = batch_size
dataloader.evaluator.dataset_name = 'bdd100k_val'
dataloader.train.dataset.names = ('bdd100k_train',)
dataloader.test.dataset.names = 'bdd100k_val'

train['output_dir'] = "./output/yolof_mask_R_50_3x"
train['max_iter'] = 90000 * 3 * default_batch_size // batch_size
train['eval_period'] = 5000 * 3 * default_batch_size // batch_size
train['init_checkpoint'] = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
train['device'] = 'cuda:0'

NUM_CLASSES = 8
model.num_classes = NUM_CLASSES
model.mask_head.num_classes = NUM_CLASSES
model.backbone.freeze_at = 2

optimizer.params.base_lr = 0.01
optimizer.lr = 0.01