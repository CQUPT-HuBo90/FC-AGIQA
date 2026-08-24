# FC-AGIQA

Official implementation of our paper project for AI-generated image quality assessment with CLIP-ConvNeXt and frequency-aware enhancement.

## Setup

Install the Python dependencies:

```bash
pip install -r AIGCIQA/AIGCIQA/requirements.txt
```

Pretrained models and image datasets are not included. The dataloaders expect the following paths:

```text
/home/dataset/AGIQA-1K/file
/home/dataset/AGIQA-3K/all
/home/dataset/AIGCIQA2023/Image
```

The five-fold CSV splits for these datasets are included under `AIGCIQA/AIGCIQA/dataloader/cv_folds/`.

## Run

### Single-fold smoke run

```bash
PYTHONPATH=./AIGCIQA python AIGCIQA/AIGCIQA/train.py \
  --benchmark AGIQA3K \
  --use_cv --fold 1 --gpu 0 \
  --num_epochs 5
```

Use `AGIQA1K`, `AGIQA3Kc`, or `AIGCIQA2023q/a/c` for the other supported tasks.

### Main experiment

Set `BENCHMARKS`, `CV_FOLDS`, `EPOCHS`, and `GPU` in `run_fmac.sh`, then run:

```bash
bash run_fmac.sh
```

The script defaults to the official AIGCIQA-20K split. To use the included CV splits, set `BENCHMARKS` to the desired dataset before running.

Training checkpoints and result files are written to the local `ckpts/` and `exp/` directories.
