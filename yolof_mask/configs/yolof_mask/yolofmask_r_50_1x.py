from yolof_mask.configs.yolof_mask.train import train
from yolof_mask.configs.yolof.optim import YOLOF_SGD as optimizer
from yolof_mask.configs.yolof.coco_schedule import lr_multiplier_1x_b16 as lr_multiplier
from yolof_mask.configs.yolof.coco_dataloader import dataloader
from yolof_mask.configs.yolof_mask.yolofmask_r_50 import model

default_batch_size = 16
batch_size = 2

dataloader.train.mapper.use_instance_mask = True
dataloader.train.mapper.instance_mask_format = "bitmask"
dataloader.train.total_batch_size = batch_size
dataloader.evaluator.dataset_name = 'bdd100k_val'
dataloader.train.dataset.names = ('bdd100k_train',)
dataloader.test.dataset.names = 'bdd100k_val'
dataloader.test.mapper.instance_mask_format = "bitmask"

train['output_dir'] = "./output/yolofmask_r_50_1x"
train['max_iter'] = 90000 * default_batch_size // batch_size
train['eval_period'] = 5000 * default_batch_size // batch_size
train['best_checkpointer']['val_metric'] = "segm/AP50"
train['init_checkpoint'] = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
train['device'] = 'cuda:0'

model.num_classes = 8
model.yolof.num_classes = 8
model.mask_head.num_classes = 8
model.yolof.backbone.freeze_at = 2

optimizer.params.base_lr = 0.01
optimizer.lr = 0.01