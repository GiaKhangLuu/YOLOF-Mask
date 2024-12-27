# YOLOF Mask

This model trained and evaluted on two public datasets: COCO2017 and BDD100K

## Result sample

BDD100K results

![BDD100K result](./asset/bdd100k_image.png)

COCO2017 result

![COCO result](./asset/coco_output.png)

## Instance segmentation results on COCO2017 test

| Config | Lr sched | Mask mAP (%) | #params (m) | download |
|:---:|:---:|:---:|:---:|:---:|
| [ResNet_50](./configs/InstanceSegmentation/yolof_mask_R_50_3x.py) | 3x | 32.4 | 46.8 | [pth](https://drive.google.com/file/d/1rHY5007rUm_QyZKnnVUVrRfXCffLSfR6/view?usp=drive_link) |
| [ResNet_50_SE_SAM](./configs/InstanceSegmentation/yolof_mask_R_50_SE_SAM_3x.py) | 3x | 34.7 | 49.3 | [pth](https://drive.google.com/file/d/139dUcXqmjrGCCXedviO19sFUy55TQJRB/view?usp=drive_link) |
| [RegNetX_4gf](./configs/InstanceSegmentation/yolof_mask_RegNetX_4gf_3x.py) | 3x | 35.5 | 43.7 | [pth](https://drive.google.com/file/d/1e72alvpqnqt2WFTdhxQpLxPL8_7N7Vs3/view?usp=drive_link) |
| [RegNetX_4gf_SAM](./configs/InstanceSegmentation/yolof_mask_RegNetX_4gf_SAM_3x.py) | 3x | 35.6 | 43.7 | [pth](https://drive.google.com/file/d/1ZyvYjnQjFgGqVt4aZCdDwYmxco4C1Tsj/view?usp=drive_link) |
| [ConvNeXt_T](./configs/InstanceSegmentation/yolof_mask_ConvNeXt_T_3x.py) | 3x | 34.7 | 50.5 | [pth](https://drive.google.com/file/d/1VEs9mGALp29vpk7ig-HC38lL77Ne3op-/view?usp=drive_link) |
| [ConvNeXt_T_SAM](./configs/InstanceSegmentation/yolof_mask_ConvNeXt_T_SAM_3x.py) | 3x | 34.8 | 50.5 | [pth](https://drive.google.com/file/d/1aLSYaFoTdR8sDPGVk2ftBGCyg7XcWBs7/view?usp=drive_link) |

## Instance segmentation results on BDD100K val

We are training!!!


## Installation

```
conda env create -f environment.yml
```

## Dataset zoo

We expect the datasets' annotations to be in COCO format to train and validate YOLOF-Mask easily. 

We currently support two datasets `BDD100K` and `COCO2017`. Download the datasets from the links below and structure the datasets as follows:

| Dataset | download |
|:---:|:---:|
| BDD100K | [link](https://drive.google.com/file/d/1U9VjD8gRFB30nh-l85cUQyRdbpsypish/view?usp=drive_link) |
| COCO2017 | [link](https://drive.google.com/file/d/1VeqhcRm4aZYiS4bWOqLtmq0pQNAkxZHR/view?usp=drive_link) |

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