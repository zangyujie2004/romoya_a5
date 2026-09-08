#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
python batch_remove_hand.py \
  --task-list /mnt/oss_data/arx3/iclr/wyt/celery/cut_celery_wyt_0902_12 \
  --task-list /mnt/oss_data/arx3/iclr/wyt/celery/cut_celery_wyt_0904_80 \
  --task-list /mnt/oss_data/arx3/iclr/wyt/celery/cut_celery_wyt_0829_13 \
  --task-list /mnt/oss_data/arx3/iclr/wyt/fakeegg/fakeegg_wyt_0903_21 \
  --task-list /mnt/oss_data/arx3/iclr/wyt/fakeegg/fakeegg_wyt_0904_9 \
  --task-list /mnt/oss_data/arx3/yz/celery/cut_celery_yz_0903_19 \
  --task-list /mnt/oss_data/arx3/yz/huanggua/peel_huanggua_coffee_up_yz_0831_12 \
  --task-list /mnt/oss_data/arx3/yz/huanggua/peel_huanggua_coffee_yz_0902_30 \
  --task-list /mnt/oss_data/arx3/yz/huanggua/peel_huanggua_coffee_yz_0902_30 \
  --task-list /mnt/oss_data/arx3/yz/huanggua/peel_huanggua_coffee_yz_0903_3 \
  --task-list /mnt/oss_data/arx3/yz/wipe_plate/wipe_plate_yz_0902_50 \
  --gpus 0,1,2,3 \
  --workers-per-gpu 3   #每张GPU处理视频数量，最多3
