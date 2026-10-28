#!/usr/bin/env bash
# Download the detector and pose-model configs + checkpoints into ./checkpoints
# using OpenMMLab's `mim` tool (installed with the OpenMMLab stack).
# Weights come from download.openmmlab.com and are released under Apache-2.0
# by OpenMMLab; they are NOT committed to this repository.
set -euo pipefail

DEST="${1:-checkpoints}"
mkdir -p "$DEST"

if ! command -v mim >/dev/null 2>&1; then
  echo "error: 'mim' not found. Install it with: pip install -U openmim" >&2
  exit 1
fi

mim download mmdet --config faster-rcnn_r50_fpn_1x_coco --dest "$DEST"
mim download mmpose --config td-hm_res50_8xb64-210e_coco-256x192 --dest "$DEST"

echo "Model files saved to $DEST:"
ls -1 "$DEST"
