#!/bin/bash
set -e

GPU_ID=$1     
SUBJ_ID=$2     
DRIVER_ID=$3

if [ -z "$GPU_ID" ] || [ -z "$SUBJ_ID" ] || [ -z "$DRIVER_ID" ]; then
    echo "Usage: bash $0 <GPU_ID> <SUBJ_ID> <DRIVER_ID>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

BASE_DATA_PATH='/PATH/TO/DATASET' # Processed nersemble
DATA_PATH="${BASE_DATA_PATH}/UNION10_${DRIVER_ID}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine" # name after v16 could be different
MODEL_PATH="XXX/TexAvatars_repo/output/${SUBJ_ID}_texavatars"

echo "Using GPU $GPU_ID for NeRSemble Subject #$SUBJ_ID and FREE Driver #$DATA_PATH"
python render_avatar.py -m ${MODEL_PATH}  -t ${DATA_PATH} --select_camera_id 8 --white_background \
    --add_tongue \
    --evaluate