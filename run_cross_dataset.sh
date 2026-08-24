#!/bin/bash
# FMAC 本模型跨库实验: train-on-source / test-on-target

set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONWARNINGS="ignore"

mkdir -p exp ckpts

GPU=${GPU:-0}
EPOCHS=${EPOCHS:-60}
SEED=${SEED:-224}
RESULTS_CSV=${RESULTS_CSV:-cross_dataset_results.csv}
RUN_ID=${RUN_ID:-"cross_iqa_stable_$(date +%Y%m%d_%H%M%S)"}

DATASETS=("AGIQA1K" "AGIQA3K" "AIGCIQA2023q")

echo "========================================"
echo "FMAC 跨库实验 - GPU ${GPU}"
echo "数据集: ${DATASETS[*]}"
echo "Run ID: ${RUN_ID}"
echo "结果CSV: ${RESULTS_CSV}"
echo "========================================"

for SOURCE in "${DATASETS[@]}"; do
    for TARGET in "${DATASETS[@]}"; do
        if [ "$SOURCE" = "$TARGET" ]; then
            continue
        fi

        LOG_INFO="cross_${SOURCE}_to_${TARGET}"
        echo "[GPU ${GPU}] Train ${SOURCE} -> Test ${TARGET}"

        PYTHONPATH=./AIGCIQA:$PYTHONPATH python AIGCIQA/AIGCIQA/train.py \
            --cross_dataset \
            --train_benchmark "$SOURCE" \
            --test_benchmark "$TARGET" \
            --backbone clipconvnext \
            --text_encoder clip_convnext \
            --num_epochs "$EPOCHS" \
            --train_batch_size 16 \
            --lr 1e-5 \
            --seed "$SEED" \
            --min_lr 1e-6 \
            --weight_decay 5e-4 \
            --gpu "$GPU" \
            --quality_preset iqa_stable \
            --cross_results_csv "$RESULTS_CSV" \
            --run_id "$RUN_ID" \
            --ablation_mode full \
            --use_frequency_features \
            --use_freq_enhanced_fusion \
            --log_info "$LOG_INFO"
        sleep 2
    done
done

echo "所有跨库实验完成"
echo "结果保存在: ${RESULTS_CSV} (run_id=${RUN_ID})"
