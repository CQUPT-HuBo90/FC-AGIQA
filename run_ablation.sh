#!/bin/bash
# 消融实验 - 质量评价任务，5折CV
# 默认不跑full；正式表格消融为 clip_only、w/o AFF、w/o MFB。

set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONWARNINGS="ignore"

mkdir -p exp ckpts

GPU=${GPU:-0}
EPOCHS=${EPOCHS:-60}
SEED=${SEED:-224}
RESULTS_CSV=${RESULTS_CSV:-ablation_results.csv}
RUN_ID=${RUN_ID:-"ablation_iqa_stable_$(date +%Y%m%d_%H%M%S)"}
DATASET_LIST=${DATASET_LIST:-"AGIQA3K AGIQA1K AIGCIQA2023q"}
FOLD_LIST=${FOLD_LIST:-"1 2 3 4 5"}
ABLATION_LIST=${ABLATION_LIST:-"clip_only no_freq_clip_fusion no_mfb"}

read -r -a DATASETS <<< "$DATASET_LIST"
read -r -a CV_FOLDS <<< "$FOLD_LIST"
read -r -a ABLATIONS <<< "$ABLATION_LIST"

echo "========================================"
echo "完整消融实验 - GPU ${GPU}"
echo "数据集: ${DATASETS[*]}"
echo "模式: ${ABLATIONS[*]}"
echo "Folds: ${CV_FOLDS[*]}"
echo "Run ID: ${RUN_ID}"
echo "结果CSV: ${RESULTS_CSV}"
echo "========================================"

build_ablation_args() {
    local ablation="$1"
    case "$ablation" in
        "baseline"|"clip_only")
            echo "--ablation_mode baseline"
            ;;
        "full")
            echo "--ablation_mode full --use_frequency_features --use_freq_enhanced_fusion"
            ;;
        "no_freq_encoder")
            echo "--ablation_mode no_freq_encoder"
            ;;
        "no_freq_clip_fusion")
            echo "--ablation_mode no_freq_clip_fusion --use_frequency_features --use_freq_enhanced_fusion --disable_freq_clip_fusion"
            ;;
        "no_mfb")
            echo "--ablation_mode no_mfb --use_frequency_features --use_freq_enhanced_fusion --disable_mfb_fusion"
            ;;
        *)
            echo "未知消融模式: $ablation" >&2
            return 1
            ;;
    esac
}

for DATASET in "${DATASETS[@]}"; do
    for ABLATION in "${ABLATIONS[@]}"; do
        ABLATION_ARGS=$(build_ablation_args "$ABLATION")
        for FOLD in "${CV_FOLDS[@]}"; do
            LOG_INFO="ablation_${ABLATION}_${DATASET}_fold${FOLD}"
            echo "[GPU ${GPU}] ${DATASET} - ${ABLATION} - Fold ${FOLD}"

            PYTHONPATH=./AIGCIQA:$PYTHONPATH python AIGCIQA/AIGCIQA/train.py \
                --benchmark "$DATASET" \
                --backbone clipconvnext \
                --text_encoder clip_convnext \
                --num_epochs "$EPOCHS" \
                --train_batch_size 16 \
                --lr 1e-5 \
                --seed "$SEED" \
                --min_lr 1e-6 \
                --weight_decay 5e-4 \
                --gpu "$GPU" \
                --use_cv \
                --fold "$FOLD" \
                --quality_preset iqa_stable \
                --results_csv "$RESULTS_CSV" \
                --run_id "$RUN_ID" \
                $ABLATION_ARGS \
                --log_info "$LOG_INFO"
            sleep 2
        done
    done
done

echo "所有消融实验完成"
echo "结果保存在: ${RESULTS_CSV} (run_id=${RUN_ID})"
