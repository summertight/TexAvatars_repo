#!/bin/bash
set -e

GPU_ID=$1    
SUBJ_ID=$2   

if [ -z "$GPU_ID" ] || [ -z "$SUBJ_ID" ]; then
    echo "Usage: bash $0 <GPU_ID> <SUBJ_ID>"
    exit 1
fi


export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Using GPU $GPU_ID for NeRSemble Subject #$SUBJ_ID"

BASE_DATA_PATH='/PATH/TO/DATASET' # Processed nersemble
DATA_PATH="${BASE_DATA_PATH}/UNION10_${SUBJ_ID}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine" # name after v16 could be different
MODEL_PATH="XXX/TexAvatars_repo/output/${SUBJ_ID}_texavatars"

PORT=$((49152 + $(od -An -N2 -i /dev/urandom) % 16384))
# echo "Randomly selected PORT: $PORT"
# echo $RANDOM
echo "Using port $PORT for NeRSemble Subject #$SUBJ_ID"
python train.py \
    -s ${DATA_PATH} \
    -m ${MODEL_PATH} \
    --eval --on_top --white_background --port $PORT \
    --add_tongue \
    --lambda_vgg 0.01 --vgg_kickin 300000 \
