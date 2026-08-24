#!/bin/bash
# GPU 0: clip_only (w/o Frequency-aware module)

set -e

export CUDA_VISIBLE_DEVICES=0
export GPU=${GPU:-0}
export ABLATION_LIST=${ABLATION_LIST:-"clip_only"}

bash run_ablation.sh
