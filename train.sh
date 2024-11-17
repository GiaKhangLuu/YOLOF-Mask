WORK_DIR=$(pwd)
DATASET_NAME=bdd100k
TASK=ins_seg

IMG_DIR=${WORK_DIR}/${DATASET_NAME}/images/10k
ANNOT_DIR=${WORK_DIR}/${DATASET_NAME}/labels_coco/${TASK}
CONFIG_FILE=${WORK_DIR}/configs/InstanceSegmentation/yolof_mask_R_50_SE_1x.py

python3 train.py -c ${CONFIG_FILE} \
 --image_dir ${IMG_DIR} \
 --annot_dir ${ANNOT_DIR} \
 --task ${TASK} \
 --dataset_name ${DATASET_NAME}