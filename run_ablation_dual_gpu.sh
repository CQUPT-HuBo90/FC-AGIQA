#!/bin/bash
# 双卡并行消融：GPU0跑 clip_only，GPU1跑 no_freq_clip_fusion/no_mfb。

set -e

RUN_ID=${RUN_ID:-"ablation_iqa_stable_$(date +%Y%m%d_%H%M%S)"}
RESULTS_CSV=${RESULTS_CSV:-ablation_results.csv}
EPOCHS=${EPOCHS:-60}
SEED=${SEED:-224}
DATASET_LIST=${DATASET_LIST:-"AGIQA3K AGIQA1K AIGCIQA2023q"}
FOLD_LIST=${FOLD_LIST:-"1 2 3 4 5"}

echo "========================================"
echo "双卡消融实验"
echo "GPU0: clip_only (w/o Frequency-aware module)"
echo "GPU1: no_freq_clip_fusion no_mfb"
echo "Run ID: ${RUN_ID}"
echo "结果CSV: ${RESULTS_CSV}"
echo "========================================"

GPU=0 \
RUN_ID="$RUN_ID" \
RESULTS_CSV="$RESULTS_CSV" \
EPOCHS="$EPOCHS" \
SEED="$SEED" \
DATASET_LIST="$DATASET_LIST" \
FOLD_LIST="$FOLD_LIST" \
ABLATION_LIST="clip_only" \
bash run_ablation.sh &
PID0=$!

GPU=1 \
RUN_ID="$RUN_ID" \
RESULTS_CSV="$RESULTS_CSV" \
EPOCHS="$EPOCHS" \
SEED="$SEED" \
DATASET_LIST="$DATASET_LIST" \
FOLD_LIST="$FOLD_LIST" \
ABLATION_LIST="no_freq_clip_fusion no_mfb" \
bash run_ablation.sh &
PID1=$!

set +e
wait "$PID0"
STATUS0=$?
wait "$PID1"
STATUS1=$?
set -e

if [ "$STATUS0" -ne 0 ] || [ "$STATUS1" -ne 0 ]; then
    echo "双卡消融有任务失败: GPU0=${STATUS0}, GPU1=${STATUS1}" >&2
    exit 1
fi

echo "双卡消融实验完成"
echo "结果保存在: ${RESULTS_CSV} (run_id=${RUN_ID})"
