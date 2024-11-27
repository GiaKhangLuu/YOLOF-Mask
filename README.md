# YOLOF Mask

## Instance segmentation results on COCO2017 test

We are training!!!

## Instance segmentation results on BDD100K val

| Config | Lr sched | Box mAP | Mask mAP | #params | FLOPs | download |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [ResNet_50_SE](./configs/InstanceSegmentation/yolof_mask_R_50_SE_3x.py) | 3x | 22.0 | 17.57 | ... | ... | [pth](https://drive.google.com/file/d/1bEfqB9SqwyMNAWOXQALvWY5ZFVnr0TY8/view?usp=drive_link) |
| [RegNetX_4gf](./configs/InstanceSegmentation/yolof_mask_RegNetX_4gf_3x.py) | 3x | 24.39 | 20.02 | ... | ... | [pth](https://drive.google.com/file/d/1MQC-nyKCa3C9Pz7p4MFE8cbGJuBynHKT/view?usp=drive_link) | 
||

## Installation

```
conda env create -f environment.yml
```

## Dataset zoo

We expect the datasets' annotations to be in COCO format to train and validate YOLOF-Mask easily. 

We currently support two datasets `BDD100K` and `COCO2017`. Download the datasets from the links below and structure the datasets as follows:

| Dataset | download |
|:---:|:---:|
| BDD100K | ... |
| COCO2017 | [link](https://drive.google.com/file/d/1VeqhcRm4aZYiS4bWOqLtmq0pQNAkxZHR/view?usp=drive_link) |
||

## Training 

```bash
export WORK_DIR=$(pwd)
export DATASET_NAME=bdd100k
export TASK=ins_seg

export IMG_DIR=${WORK_DIR}/dataset_zoo/${DATASET_NAME}/images/10k
export ANNOT_DIR=${WORK_DIR}/dataset_zoo/${DATASET_NAME}/labels_coco/${TASK}
export CONFIG_FILE=${WORK_DIR}/configs/InstanceSegmentation/yolof_mask_ConvNeXt_T_1x.py

python3 train.py -c ${CONFIG_FILE} \
 --image_dir ${IMG_DIR} \
 --annot_dir ${ANNOT_DIR} \
 --task ${TASK} \
 --dataset_name ${DATASET_NAME} \
 --resume
```

## Evaluation

```bash
export WORK_DIR=$(pwd)
export DATASET_NAME=bdd100k
export TASK=ins_seg

export IMG_DIR=${WORK_DIR}/dataset_zoo/${DATASET_NAME}/images/10k
export ANNOT_DIR=${WORK_DIR}/dataset_zoo/${DATASET_NAME}/labels_coco/${TASK}
export CONFIG_FILE=${WORK_DIR}/configs/InstanceSegmentation/yolof_mask_regnetx_4gf_1x.py
export MODEL_WEIGHT=${WORK_DIR}/output/yolof_mask_RegNetX_4gf_1x/model_best.pth

python3 evaluate.py -c ${CONFIG_FILE} \
 --image_dir ${IMG_DIR} \
 --annot_dir ${ANNOT_DIR} \
 --task ${TASK} \
 --dataset_name ${DATASET_NAME} \
 --model_weight ${MODEL_WEIGHT} \
 --score_threshold 0.5 \
 --max_dets_per_image 200
```

## Inference

To use YOLOF_MASK model, you can predict by CLI or Python code.

### CLI

```bash
python3 infer.py -c ./configs/InstanceSegmentation/yolof_mask_R_50_SE_3x.py \
    --device 0 \
    --input_path ./test/data/bdd100k/acaaf824-00000000.jpg \
    --output_path ./output/inference/image_result.jpg \
    --model_weight /path/to/model.pth \
    --score_threshold 0.5 \
    --max_dets 100 \
    --dataset_name bdd100k \
    --task ins_seg \
    --vis_result
```

### Python code
```python
import cv2
import matplotlib.pyplot as plt

from detectron2.config import LazyConfig
from detectron2.checkpoint import DetectionCheckpointer

from yolof_mask.engine.default_predictor import DefaultPredictor
from yolof_mask.utils.visualizer import Visualizer, ColorMode

config_path = './configs/InstanceSegmentation/yolof_mask_RegNetX_4gf_3x.py'
cfg = LazyConfig.load(config_path)
cfg.train.device = "cuda:0"
cfg.train.init_checkpoint = "./output/yolof_mask_RegNetX_4gf_3x/model_best.pth"

predictor = DefaultPredictor(cfg)

img = cv2.imread("./test/data/bdd100k/acaaf824-00000000.jpg")
predictions = predictor(img)

# Currently, we just support for bdd100k
dataset_name = "bdd100k"
if dataset_name == "bdd100k":
    class_names = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
v = Visualizer(img, class_names=class_names)
out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
img_result = out.get_image()[..., ::-1]

_ = plt.figure(figsize=(12, 10))
plt.imshow(img_result)
```

## Citation

...