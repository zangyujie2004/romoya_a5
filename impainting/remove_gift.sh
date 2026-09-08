#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python batch_remove_hand.py \
  --task-list /mnt/oss_data/arx2/zyp/zyp_mofang_0814_50 \
  --task-list /mnt/oss_data/arx2/zyp/zyp_lemon_0814_50 \
  --task-list /mnt/oss_data/arx2/zyp/zyp_shiguan_0814_100 \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers-per-gpu 4
