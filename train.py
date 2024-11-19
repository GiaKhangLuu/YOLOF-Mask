import argparse
import os 
import detectron2
from pathlib import Path

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets.coco_panoptic import register_coco_panoptic_separated 
from detectron2.data.datasets import register_coco_instances
from detectron2.config import LazyConfig
from detectron2.engine import default_setup

from tools.lazyconfig_train_net import do_train

def main(**kwargs):
    config_path = kwargs.get("config")
    task = kwargs.get("task")
    dataset_name = kwargs.get("dataset_name")
    img_dir = kwargs.get("image_dir")
    annot_dir = kwargs.get("annot_dir")
    is_resume = kwargs.get("resume")

    assert dataset_name in ["bdd100k"]
    assert task in ["ins_seg", "panoptic"]

    for split in ["train", "val"]:
        d_name = dataset_name + f"_{split}"
        img_phase_dir = os.path.join(img_dir, split)
        annot_phase_path = os.path.join(annot_dir, f"ins_seg_{split}_coco.json")

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
        num_gpus=1
        num_machines=1
        resume=is_resume

    args = Args()

    cfg = LazyConfig.load(config_path)

    default_setup(cfg, args)

    do_train(args, cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using a configuration file.")
    parser.add_argument(
        "-c", 
        "--config", 
        type=str, 
        required=True, 
        help="Path to the configuration file."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Image directory."
    )
    parser.add_argument(
        "--annot_dir",
        type=str,
        required=True,
        help="Annotation directory."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="ins_seg",
        required=True,
        help="Task to train."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="bdd100k",
        required=True,
        help="Name of data to train."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Flag to resume training."
    )

    args = parser.parse_args()
    main(**vars(args))