export WORK_DIR=$(pwd)
export BATCH_SIZE=4
export CUDA_DEVICE=0
export CONFIG_FILE=${WORK_DIR}/configs/bdd100k/yolof_mask_RegNetX_4gf_SAM_3x.py
export OUTPUT_DIR=./output/bdd100k/yolof_mask_RegNetY_4gf_SAM_3x

python3 train.py -c ${CONFIG_FILE} \
 --batch_size ${BATCH_SIZE} \
 --device ${CUDA_DEVICE} \
 --output_dir ${OUTPUT_DIR}
