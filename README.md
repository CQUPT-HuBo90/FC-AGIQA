# FC-AGIQA

论文项目代码：基于 CLIP-ConvNeXt 和频域增强的 AI 生成图像质量评价。

## 环境

```bash
pip install -r AIGCIQA/AIGCIQA/requirements.txt
```

预训练模型和数据集不会随仓库提供。默认数据路径为：

```text
/home/dataset/AGIQA-1K/file
/home/dataset/AGIQA-3K/all
/home/dataset/AIGCIQA2023/Image
```

三组 5 折划分已放在 `AIGCIQA/AIGCIQA/dataloader/cv_folds/`。

## 运行

### 单折 smoke run

```bash
PYTHONPATH=./AIGCIQA python AIGCIQA/AIGCIQA/train.py \
  --benchmark AGIQA3K \
  --use_cv --fold 1 --gpu 0 \
  --num_epochs 5
```

可将 `--benchmark` 改为 `AGIQA1K`、`AGIQA3Kc` 或 `AIGCIQA2023q/a/c`。

### 主实验

编辑 `run_fmac.sh` 中的 `BENCHMARKS`、`CV_FOLDS`、`EPOCHS` 和 `GPU`，然后运行：

```bash
bash run_fmac.sh
```

脚本默认配置为 AIGCIQA-20K 官方划分；运行三组 CV 数据集时，请将 `BENCHMARKS` 改为相应名称。

### 消融实验

```bash
GPU=0 EPOCHS=5 FOLD_LIST="1" bash run_ablation.sh
```

也可以使用 `run_clip_only.sh`、`run_ablation_gpu0.sh` 或 `run_ablation_gpu1.sh`。

## 结果分析

训练结果默认写入 `ckpts/` 和 CSV 文件。5 折结果可用以下命令汇总：

```bash
python fast_cv_analyzer.py --dataset AGIQA3K --save
```

跨数据集实验使用：

```bash
bash run_cross_dataset.sh
```
