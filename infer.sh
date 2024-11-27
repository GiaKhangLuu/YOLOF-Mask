WORK_DIR=$(pwd)
DATASET_NAME=bdd100k
TASK=ins_seg

CONFIG_FILE=${WORK_DIR}/configs/InstanceSegmentation/yolof_mask_RegNetX_4gf_1x.py
MODEL_WEIGHT=${WORK_DIR}/output/yolof_mask_RegNetX_4gf_1x/model_best.pth
INPUT_PATH=/home/giakhang/dev/multi_task_autopilot/YOLOF-Mask/dataset_zoo/bdd100k/images/10k/test/ace9bf57-669189d2.jpg
OUTPUT_PATH=${WORK_DIR}/output/inference/result.jpg

python3 infer.py -c ${CONFIG_FILE} \
 --device 0 \
 --input_path ${INPUT_PATH} \
 --output_path ${OUTPUT_PATH} \
 --model_weight ${MODEL_WEIGHT} \
 --task ${TASK} \
 --dataset_name ${DATASET_NAME} \
 --score_threshold 0.5 \
 --vis_result