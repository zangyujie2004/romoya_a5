#!/usr/bin/env bash
set -euo pipefail

original_batch_pid=1958128
retry_log=/mnt/workspace/zyj/preprocess/impainting/codex_retry_after_pi05_20260819.log

exec >>"${retry_log}" 2>&1
echo "[$(date -Is)] waiting for original batch pid ${original_batch_pid}"
while kill -0 "${original_batch_pid}" 2>/dev/null; do
    sleep 30
done

echo "[$(date -Is)] original batch finished; scanning and retrying incomplete episodes"
cd /mnt/workspace/zyj/preprocess/impainting
exec /mnt/data/lcy/miniconda3/envs/zangyj/bin/python batch_remove_hand.py \
    --task-list /mnt/oss_data/arx3/iclr/wl/egg/fakeegg_fial_wl_0819_50 \
    --gpus 0,1,2,3 \
    --workers-per-gpu 3
