#!/bin/bash
# GPU 1: no_freq_clip_fusion + no_mfb

set -e

export CUDA_VISIBLE_DEVICES=1
export GPU=${GPU:-1}
export ABLATION_LIST=${ABLATION_LIST:-"no_freq_clip_fusion no_mfb"}

bash run_ablation.sh
