export WORK_DIR=$(pwd)
export DATASET_NAME=coco2017
export TASK=ins_seg
export NUM_CLASSES=80
export BATCH_SIZE=4
export CUDA_DEVICE=0
export OUTPUT_DIR=./output/yolof_mask_RegNetY_4gf_SAM_3x

export IMG_DIR=${WORK_DIR}/datasets/${DATASET_NAME}/coco2017
export ANNOT_DIR=${WORK_DIR}/datasets/${DATASET_NAME}/coco2017/annotations
export CONFIG_FILE=${WORK_DIR}/configs/InstanceSegmentation/yolof_mask_RegNetY_4gf_SAM_3x.py

python3 train.py -c ${CONFIG_FILE} \
 --image_dir ${IMG_DIR} \
 --annot_dir ${ANNOT_DIR} \
 --task ${TASK} \
 --dataset_name ${DATASET_NAME} \
 --num_classes ${NUM_CLASSES} \
 --batch_size ${BATCH_SIZE} \
 --device ${CUDA_DEVICE} \
 --output_dir ${OUTPUT_DIR}
