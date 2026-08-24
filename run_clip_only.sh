#!/bin/bash
# Clip-only baseline: CLIP-ConvNeXt image/text features only, no frequency branch, no AFF, no MFB.

set -e

export GPU=${GPU:-0}
export ABLATION_LIST="clip_only"
export RESULTS_CSV=${RESULTS_CSV:-clip_only_results.csv}
export RUN_ID=${RUN_ID:-"clip_only_iqa_stable_$(date +%Y%m%d_%H%M%S)"}

bash run_ablation.sh
