#!/bin/bash

# 设置离线模式，避免网络连接问题
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONWARNINGS="ignore"
export TRANSFORMERS_VERBOSITY=error
# 限制每个训练进程的CPU线程数，避免多GPU并行训练时CPU线程过度订阅导致GPU等数据
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-2}
# 注意: CUDA_LAUNCH_BLOCKING=1 仅用于调试(强制同步执行)，会显著拖慢训练，常规训练务必关闭

# 创建必要的目录
mkdir -p exp
mkdir -p ckpts

# ========== 训练参数配置 ==========
# 支持: "AGIQA1K" "AGIQA3K" "AIGCIQA2023q/a/c" "AGIQA3Kc" "I2IQAq/a/c" "AIGCIQA20K"
# I2IQAq=质量, I2IQAa=真实性, I2IQAc=一致性；先默认跑新数据集质量任务
BENCHMARKS=("AIGCIQA20K")

# ========== 交叉验证配置 ==========
USE_CV=true                             # 启用交叉验证模式 (true/false)
CV_FOLDS=(1 2 3 4 5)                    # 完整5折（单折试训可只保留一个fold）
BACKBONE="clipconvnext"                 # clipconvnext
TEXT_ENCODER="clip_convnext"
EPOCHS=60                               # 使用--num_epochs参数
BATCH_SIZE=16                           # 使用--train_batch_size参数
LR=1e-5                                 # 学习率
GPU=0
SEED=224                                # 随机种子
SAVE_BEST_START_EPOCH=9                 # 从第N个epoch开始保存最优模型（避免第1个epoch保存）
RUN_ID="run_fmac_$(date +%m%d_%H%M%S)_gpu${GPU//,/-}"
RESULTS_CSV="ablation_results.csv"
LOG_FILE="exp/FMAC_gpu${GPU//,/-}.log"

# ========== 学习率调度器关键参数 ==========
SCHEDULER="cosine"                      # 稳定默认: 余弦退火
MIN_LR=1e-6                             # 最小学习率（余弦退火的最低点）
T_MAX=60                                # 余弦退火周期
WARMUP_EPOCHS=5                         # 仅在SCHEDULER=warmup_cosine时生效
TEXT_LR_FACTOR=1                        # 文本编码器学习率因子
IMAGE_LR_FACTOR=0.2                     # 质量/真实性任务默认85配置: image encoder用0.2x LR，head/fusion/text保持1.0x
WEIGHT_DECAY=5e-4                       # 权重衰减

# ========== 一致性任务对比损失参数 ==========
GLOBAL_CONTRASTIVE_WEIGHT=0.1

# ========== 质量/真实性任务默认85配置 ==========
# 已在AIGCIQA2023q 5折验证: label_norm=none, hard rank, image_lr_factor=0.2, text不冻结
QUALITY_PRESET="iqa_stable"             # none / iqa_stable；仅质量评价任务自动启用
TRAIN_AUG_POLICY="legacy"               # legacy / quality_stable / correspondence_stable
USE_EMA=false                           # 峰值协议默认关闭EMA
EMA_DECAY=0.999
LOSS_SMOOTH_L1_WEIGHT=0.55
LOSS_PEARSON_WEIGHT=0.20
LOSS_RANK_WEIGHT=0.25
RANK_MIN_DIFF=0.03
RANK_TEMPERATURE=0.5
RANK_LOSS_TYPE="hard"                   # hard / soft_weighted
LABEL_NORM="none"                       # none / train_zscore
FREEZE_TEXT_ENCODER=false               # 85配置保持文本编码器可训练
LOG_TRAINING_DIAGNOSTICS=false          # 需要诊断时改为true

# ========== 稳定性优化参数 ==========
GRADIENT_ACCUMULATION=1                 # 每个批次都更新梯度

is_official_split_dataset() {
    [[ "$1" == "AIGCIQA20K" ]]
}

is_quality_preset_dataset() {
    [[ "$1" == "AGIQA1K" || "$1" == "AGIQA3K" || "$1" == "AIGCIQA2023q" || "$1" == "I2IQAq" || "$1" == "AIGCIQA20K" ]]
}

echo "全局配置:"
echo "数据集列表: ${BENCHMARKS[*]}"
echo "模型: $BACKBONE + $TEXT_ENCODER"
echo "训练轮数: $EPOCHS"
echo "批次大小: $BATCH_SIZE"
echo "基础学习率: $LR"
echo "随机种子: $SEED"
echo "GPU: $GPU"
echo "运行标识(RUN_ID): $RUN_ID"
echo "结果CSV: $RESULTS_CSV"
echo "日志文件: $LOG_FILE"
echo "保存最佳模型起始epoch: $SAVE_BEST_START_EPOCH"
echo ""
echo "交叉验证配置:"
echo "启用交叉验证: $USE_CV"
if [ "$USE_CV" = true ]; then
    echo "要运行的folds: ${CV_FOLDS[*]}"
else
    echo "使用8:2随机划分模式"
fi
echo ""
echo "学习率调度器配置:"
echo "调度器类型: $SCHEDULER"
echo "最小学习率: $MIN_LR"
echo "余弦退火周期: $T_MAX"
echo "预热轮数: $WARMUP_EPOCHS"
echo "文本编码器学习率因子: $TEXT_LR_FACTOR"
echo "图像编码器学习率因子: $IMAGE_LR_FACTOR"
echo "权重衰减: $WEIGHT_DECAY"
echo ""
echo "全数据集平均优化配置:"
echo "质量任务预设: $QUALITY_PRESET"
echo "训练增强策略: $TRAIN_AUG_POLICY"
echo "EMA: $USE_EMA (decay=$EMA_DECAY)"
echo "损失权重: SmoothL1=$LOSS_SMOOTH_L1_WEIGHT, Pearson=$LOSS_PEARSON_WEIGHT, Rank=$LOSS_RANK_WEIGHT"
echo "排序损失: type=$RANK_LOSS_TYPE, min_diff=$RANK_MIN_DIFF, temperature=$RANK_TEMPERATURE"
echo "标签归一化: $LABEL_NORM"
echo "冻结文本编码器: $FREEZE_TEXT_ENCODER"
echo "训练诊断日志: $LOG_TRAINING_DIAGNOSTICS"
echo ""
echo "稳定性优化配置:"
echo "优化器: AdamW (已升级，提供更好的权重衰减)"
echo "梯度累积步数: $GRADIENT_ACCUMULATION (=1表示标准训练，适合小数据集)"
echo "梯度裁剪阈值: 1.0 (更保守的设置)"
echo "过拟合检测: 已禁用 (完整训练所有epochs)"
echo "========================================"

# 记录开始时间
START_TIME=$(date)
echo "开始时间: $START_TIME"

# ========== 循环运行所有数据集和fold ==========
if [ "$USE_CV" = true ]; then
    TOTAL_RUNS=0
    for BENCHMARK in "${BENCHMARKS[@]}"; do
        if is_official_split_dataset "$BENCHMARK"; then
            TOTAL_RUNS=$((TOTAL_RUNS + 1))
        else
            TOTAL_RUNS=$((TOTAL_RUNS + ${#CV_FOLDS[@]}))
        fi
    done
    echo "交叉验证/官方划分混合模式: 将运行 $TOTAL_RUNS 次训练"
else
    TOTAL_RUNS=${#BENCHMARKS[@]}
    echo "非CV/官方划分模式: 将运行 $TOTAL_RUNS 个数据集"
fi

CURRENT_COUNT=0

for BENCHMARK in "${BENCHMARKS[@]}"; do
    if [ "$USE_CV" = true ] && ! is_official_split_dataset "$BENCHMARK"; then
        # 交叉验证模式：为每个数据集运行所有fold
        for FOLD in "${CV_FOLDS[@]}"; do
            CURRENT_COUNT=$((CURRENT_COUNT + 1))
            LOG_INFO="run_fmac_cv_${BENCHMARK}_fold${FOLD}_$(date +%m%d_%H%M)"

            EXTRA_CONTRASTIVE_ARGS=""
            if [[ "$BENCHMARK" == "AIGCIQA2023c" || "$BENCHMARK" == "AGIQA3Kc" || "$BENCHMARK" == "I2IQAc" ]]; then
                EXTRA_CONTRASTIVE_ARGS="--use_global_contrastive --global_contrastive_weight $GLOBAL_CONTRASTIVE_WEIGHT"
            fi

            EXTRA_EMA_ARGS=""
            if [ "$USE_EMA" = true ]; then
                EXTRA_EMA_ARGS="--use_ema --ema_decay $EMA_DECAY"
            fi

            EXTRA_TEXT_ARGS=""
            if [ "$FREEZE_TEXT_ENCODER" = true ]; then
                EXTRA_TEXT_ARGS="--freeze_text_encoder"
            fi

            EXTRA_DIAG_ARGS=""
            if [ "$LOG_TRAINING_DIAGNOSTICS" = true ]; then
                EXTRA_DIAG_ARGS="--log_training_diagnostics"
            fi

            EXTRA_QUALITY_PRESET_ARGS=""
            if is_quality_preset_dataset "$BENCHMARK"; then
                EXTRA_QUALITY_PRESET_ARGS="--quality_preset $QUALITY_PRESET"
            fi
            
            echo ""
            echo "========================================"
            echo "开始训练 [$CURRENT_COUNT/$TOTAL_RUNS]: $BENCHMARK - Fold $FOLD"
            echo "日志标识: $LOG_INFO"
            echo "使用随机种子: $SEED"
            echo "交叉验证fold: $FOLD"
            echo "学习率配置: LR=$LR, Text_Factor=$TEXT_LR_FACTOR, Image_Factor=$IMAGE_LR_FACTOR"
            echo "========================================"
            
            # 构建交叉验证训练命令
            CMD="PYTHONPATH=./AIGCIQA:$PYTHONPATH python AIGCIQA/AIGCIQA/train.py \
                --benchmark $BENCHMARK \
                --backbone $BACKBONE \
                --text_encoder $TEXT_ENCODER \
                --num_epochs $EPOCHS \
                --train_batch_size $BATCH_SIZE \
                --lr $LR \
                --seed $SEED \
                --scheduler $SCHEDULER \
                --min_lr $MIN_LR \
                --t_max $T_MAX \
                --warmup_epochs $WARMUP_EPOCHS \
                --text_lr_factor $TEXT_LR_FACTOR \
                --image_lr_factor $IMAGE_LR_FACTOR \
                --weight_decay $WEIGHT_DECAY \
                --train_aug_policy $TRAIN_AUG_POLICY \
                --loss_smooth_l1_weight $LOSS_SMOOTH_L1_WEIGHT \
                --loss_pearson_weight $LOSS_PEARSON_WEIGHT \
                --loss_rank_weight $LOSS_RANK_WEIGHT \
                --rank_min_diff $RANK_MIN_DIFF \
                --rank_temperature $RANK_TEMPERATURE \
                --rank_loss_type $RANK_LOSS_TYPE \
                --label_norm $LABEL_NORM \
                $EXTRA_QUALITY_PRESET_ARGS \
                --save_best_start_epoch $SAVE_BEST_START_EPOCH \
                --use_frequency_features \
                --use_freq_enhanced_fusion \
                --ablation_mode full \
                $EXTRA_CONTRASTIVE_ARGS \
                $EXTRA_EMA_ARGS \
                $EXTRA_TEXT_ARGS \
                $EXTRA_DIAG_ARGS \
                --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
                --gpu $GPU \
                --results_csv \"$RESULTS_CSV\" \
                --run_id \"$RUN_ID\" \
                --log_info \"$LOG_INFO\" \
                --use_cv \
                --fold $FOLD"

            echo "执行命令:"
            echo $CMD
            echo "----------------------------------------"

            # 执行训练
            eval $CMD

            # 检查训练是否成功
            if [ $? -eq 0 ]; then
                echo "✅ $BENCHMARK Fold $FOLD 训练成功完成！"
                echo "模型文件保存在: ckpts/"
            else
                echo "❌ $BENCHMARK Fold $FOLD 训练失败！"
                echo "错误日志: $LOG_FILE"
                
                # 询问是否继续下一个fold
                echo "是否继续训练下一个fold？(y/n)"
                read -r -t 30 continue_choice
                if [[ ! $continue_choice =~ ^[Yy] ]]; then
                    echo "用户选择停止训练。"
                    exit 1
                fi
            fi
            
            echo "----------------------------------------"
            sleep 2
        done
    else
        # 非CV模式或官方划分模式
        CURRENT_COUNT=$((CURRENT_COUNT + 1))
        LOG_INFO="run_fmac_${BENCHMARK}_$(date +%m%d_%H%M)"

        EXTRA_CONTRASTIVE_ARGS=""
        if [[ "$BENCHMARK" == "AIGCIQA2023c" || "$BENCHMARK" == "AGIQA3Kc" || "$BENCHMARK" == "I2IQAc" ]]; then
            EXTRA_CONTRASTIVE_ARGS="--use_global_contrastive --global_contrastive_weight $GLOBAL_CONTRASTIVE_WEIGHT"
        fi

        EXTRA_EMA_ARGS=""
        if [ "$USE_EMA" = true ]; then
            EXTRA_EMA_ARGS="--use_ema --ema_decay $EMA_DECAY"
        fi

        EXTRA_TEXT_ARGS=""
        if [ "$FREEZE_TEXT_ENCODER" = true ]; then
            EXTRA_TEXT_ARGS="--freeze_text_encoder"
        fi

        EXTRA_DIAG_ARGS=""
        if [ "$LOG_TRAINING_DIAGNOSTICS" = true ]; then
            EXTRA_DIAG_ARGS="--log_training_diagnostics"
        fi

        EXTRA_QUALITY_PRESET_ARGS=""
        if is_quality_preset_dataset "$BENCHMARK"; then
            EXTRA_QUALITY_PRESET_ARGS="--quality_preset $QUALITY_PRESET"
        fi
        
        echo ""
        echo "========================================"
        echo "开始训练 [$CURRENT_COUNT/$TOTAL_RUNS]: $BENCHMARK"
        echo "日志标识: $LOG_INFO"
        echo "使用随机种子: $SEED"
        echo "学习率配置: LR=$LR, Text_Factor=$TEXT_LR_FACTOR, Image_Factor=$IMAGE_LR_FACTOR"
        echo "========================================"
        
        # 构建训练命令 - 所有数据集使用统一的学习率参数
        CMD="PYTHONPATH=./AIGCIQA/AIGCIQA:$PYTHONPATH python AIGCIQA/AIGCIQA/train.py \
            --benchmark $BENCHMARK \
            --backbone $BACKBONE \
            --text_encoder $TEXT_ENCODER \
            --num_epochs $EPOCHS \
            --train_batch_size $BATCH_SIZE \
            --lr $LR \
            --seed $SEED \
            --scheduler $SCHEDULER \
            --min_lr $MIN_LR \
            --t_max $T_MAX \
            --warmup_epochs $WARMUP_EPOCHS \
            --text_lr_factor $TEXT_LR_FACTOR \
            --image_lr_factor $IMAGE_LR_FACTOR \
            --weight_decay $WEIGHT_DECAY \
            --train_aug_policy $TRAIN_AUG_POLICY \
            --loss_smooth_l1_weight $LOSS_SMOOTH_L1_WEIGHT \
            --loss_pearson_weight $LOSS_PEARSON_WEIGHT \
            --loss_rank_weight $LOSS_RANK_WEIGHT \
            --rank_min_diff $RANK_MIN_DIFF \
            --rank_temperature $RANK_TEMPERATURE \
            --rank_loss_type $RANK_LOSS_TYPE \
            --label_norm $LABEL_NORM \
            $EXTRA_QUALITY_PRESET_ARGS \
            --save_best_start_epoch $SAVE_BEST_START_EPOCH \
            --use_frequency_features \
            --use_freq_enhanced_fusion \
            --ablation_mode full \
            $EXTRA_CONTRASTIVE_ARGS \
            $EXTRA_EMA_ARGS \
            $EXTRA_TEXT_ARGS \
            $EXTRA_DIAG_ARGS \
            --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
            --gpu $GPU \
            --results_csv \"$RESULTS_CSV\" \
            --run_id \"$RUN_ID\" \
            --log_info \"$LOG_INFO\""

        echo "执行命令:"
        echo $CMD
        echo "----------------------------------------"

        # 执行训练
        eval $CMD

        # 检查训练是否成功
        if [ $? -eq 0 ]; then
            echo "✅ $BENCHMARK 训练成功完成！"
            echo "模型文件保存在: ckpts/"
        else
            echo "❌ $BENCHMARK 训练失败！"
            echo "错误日志: $LOG_FILE"
            
            # 询问是否继续下一个数据集
            echo "是否继续训练下一个数据集？(y/n)"
            read -r -t 30 continue_choice
            if [[ ! $continue_choice =~ ^[Yy] ]]; then
                echo "用户选择停止训练。"
                exit 1
            fi
        fi
        
        echo "----------------------------------------"
        sleep 2
    fi
done

# ========== 训练完成总结 ==========
END_TIME=$(date)
echo ""
echo "========================================"
echo "🎉 所有训练任务完成！"
echo "========================================"
echo "开始时间: $START_TIME"
echo "结束时间: $END_TIME"
echo "训练的数据集: ${BENCHMARKS[*]}"

if [ "$USE_CV" = true ]; then
    echo "训练模式: 5折交叉验证"
    echo "运行的folds: ${CV_FOLDS[*]}"
    echo "总计训练次数: $TOTAL_RUNS (${#BENCHMARKS[@]} 数据集 × ${#CV_FOLDS[@]} folds)"
else
    echo "训练模式: 8:2随机划分"
    echo "总计数据集数量: $TOTAL_RUNS"
fi

echo "使用的随机种子: $SEED"
echo ""
echo "学习率配置总结:"
echo "- 基础学习率: $LR"
echo "- 文本编码器学习率: $LR × $TEXT_LR_FACTOR"
echo "- 图像编码器学习率: $LR × $IMAGE_LR_FACTOR"
echo "- 调度器: $SCHEDULER (T_max=$T_MAX, min_lr=$MIN_LR)"
echo "- 质量任务预设: $QUALITY_PRESET"
echo ""
echo "检查结果:"
echo "- 日志文件: $LOG_FILE"
echo "- 结果CSV: $RESULTS_CSV (run_id=$RUN_ID)"
echo "- 模型文件: ckpts/"

if [ "$USE_CV" = true ]; then
    echo ""
    echo "交叉验证结果分析："
    echo "建议查看各fold的性能指标，计算平均值和标准差"
    echo "最佳模型将根据验证集性能自动保存"
fi

echo "========================================" 
