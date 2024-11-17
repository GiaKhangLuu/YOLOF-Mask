from ..common.train import train
from ..common.yolof_optim import YOLOF_SGD as optimizer
from ..common.yolof_coco_schedule import lr_multiplier_1x_b16 as lr_multiplier
from ..common.data.coco import dataloader
from ..common.models.yolof_mask_r_50_se import model

default_batch_size = 16
batch_size = 16

dataloader.train.mapper.use_instance_mask = True
dataloader.train.mapper.instance_mask_format = "bitmask"
dataloader.train.total_batch_size = batch_size
dataloader.evaluator.dataset_name = 'bdd100k_val'
dataloader.train.dataset.names = ('bdd100k_train',)
dataloader.test.dataset.names = 'bdd100k_val'

train['output_dir'] = "./output/yolof_mask_R_50_1x"
train['max_iter'] = 90000 * default_batch_size // batch_size
train['eval_period'] = 5000 * default_batch_size // batch_size
train['init_checkpoint'] = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
train['device'] = 'cuda:0'

model.num_classes = 8

optimizer.params.base_lr = 0.01
optimizer.lr = 0.01