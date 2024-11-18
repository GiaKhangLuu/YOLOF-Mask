from ..common.train import train
from ..common.optim import SGD as optimizer
from ..common.yolof_coco_schedule import lr_multiplier_1x_b16 as lr_multiplier
from ..common.data.coco import dataloader
from ..common.models.yolof_mask_regnetx_4gf import model

default_batch_size = 16
batch_size = 8

dataloader.train.mapper.use_instance_mask = True
dataloader.train.mapper.instance_mask_format = "bitmask"
dataloader.train.total_batch_size = batch_size
dataloader.evaluator.dataset_name = 'bdd100k_val'
dataloader.train.dataset.names = ('bdd100k_train',)
dataloader.test.dataset.names = 'bdd100k_val'

train['output_dir'] = "./output/yolof_mask_regnetx_4gf_1x"
train['max_iter'] = 90000 * default_batch_size // batch_size
train['eval_period'] = 5000 * default_batch_size // batch_size
train['init_checkpoint'] = "https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906383/RegNetX-4.0GF_dds_8gpu.pyth"
train['device'] = 'cuda:0'
train['cudnn_benchmark '] = True

NUM_CLASSES = 8
model.num_classes = NUM_CLASSES
model.mask_head.num_classes = NUM_CLASSES
model.backbone.freeze_at = 2

optimizer.params.base_lr = 0.01
optimizer.lr = 0.01
optimizer.weight_decay = 5e-5