import argparse
import os 
import detectron2
from pathlib import Path

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.data.datasets.coco_panoptic import register_coco_panoptic_separated 
from detectron2.data.datasets import register_coco_instances
from detectron2.config import LazyConfig
from detectron2.engine import default_setup

from yolof_mask.tools.lazyconfig_train_net import do_train

def main(**kwargs):
    config_path = kwargs.get("config")
    task = kwargs.get("task")
    dataset_name = kwargs.get("dataset_name")
    img_dir = kwargs.get("image_dir")
    annot_dir = kwargs.get("annot_dir")
    is_resume = kwargs.get("resume")
    num_classes = int(kwargs.get("num_classes"))
    batch_size = int(kwargs.get("batch_size"))
    device = int(kwargs.get("device"))
    output_dir = kwargs.get("output_dir")

    assert dataset_name in ["bdd100k", "coco2017"]
    assert task in ["ins_seg", "panoptic"]

    for split in ["train", "val"]:
        d_name = dataset_name + f"_{split}"
        if dataset_name == "bdd100k":
            img_phase_dir = os.path.join(img_dir, split)
            annot_phase_path = os.path.join(annot_dir, f"ins_seg_{split}_coco.json")
        elif dataset_name == "coco2017":
            img_phase_dir = os.path.join(img_dir, f"{split}2017")
            annot_phase_path = os.path.join(annot_dir, f"instances_{split}2017.json")


        if task == "ins_seg":
            register_coco_instances(
                d_name, 
                {},
                annot_phase_path,
                img_phase_dir
            )
        elif task == "panoptic":
            # TODO: Implement panoptic training
            pass
        else:
            raise Exception("Invalid task!!!")

    class Args(argparse.Namespace):
        config_file=config_path
        eval_only=False
        num_gpus=kwargs.get("num_gpus", 1)
        num_machines=1
        resume=is_resume

    args = Args()

    cfg = LazyConfig.load(config_path)

    cfg.dataloader.evaluator.dataset_name = f'{dataset_name}_val'
    cfg.dataloader.train.dataset.names = (f'{dataset_name}_train',)
    cfg.dataloader.test.dataset.names = f'{dataset_name}_val'

    cfg.model.num_classes = num_classes
    cfg.model.mask_head.num_classes = num_classes

    default_batch_size = 16
    cfg.train['max_iter'] = cfg.train['max_iter'] * default_batch_size // batch_size
    cfg.train['eval_period'] = cfg.train['eval_period'] * default_batch_size // batch_size
    cfg.train['output_dir'] = output_dir 
    cfg.train['device'] = f"cuda:{str(device)}"

    cfg.dataloader.train.total_batch_size = batch_size
    cfg.dataloader.test.batch_size = batch_size
    
    cfg.lr_multiplier.batch_size = batch_size

    default_setup(cfg, args)

    do_train(args, cfg)

if __name__ == "__main__":
    # TODO: Add `num_gpus` options
    parser = argparse.ArgumentParser(description="Train a model using a configuration file.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--image_dir", type=str, required=True, help="Image directory.")
    parser.add_argument("--annot_dir", type=str, required=True, help="Annotation directory.")
    parser.add_argument("--task", type=str, default="ins_seg", required=True, help="Task to train.")
    parser.add_argument("--dataset_name", type=str, default="bdd100k", required=True, help="Name of data to train.")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", type=int, default=0, help="Device to use.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory.")
    parser.add_argument("--resume", action="store_true", help="Flag to resume training.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use.")

    args = parser.parse_args()
    main(**vars(args))
