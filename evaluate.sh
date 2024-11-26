WORK_DIR=$(pwd)
DATASET_NAME=bdd100k
TASK=ins_seg

IMG_DIR=${WORK_DIR}/dataset_zoo/${DATASET_NAME}/images/10k
ANNOT_DIR=${WORK_DIR}/dataset_zoo/${DATASET_NAME}/labels_coco/${TASK}
CONFIG_FILE=${WORK_DIR}/configs/InstanceSegmentation/yolof_mask_regnetx_4gf_1x.py
MODEL_WEIGHT=${WORK_DIR}/output/yolof_mask_RegNetX_4gf_1x/model_best.pth

python3 evaluate.py -c ${CONFIG_FILE} \
 --image_dir ${IMG_DIR} \
 --annot_dir ${ANNOT_DIR} \
 --task ${TASK} \
 --dataset_name ${DATASET_NAME} \
 --model_weight ${MODEL_WEIGHT} \
 --score_threshold 0.5 \
 --max_dets_per_image 200