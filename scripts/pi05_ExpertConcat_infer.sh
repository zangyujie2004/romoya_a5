#!/usr/bin/env bash
# Synchronous rollout with the pi0.5 expert_concat policy. Start the policy server first
# (tactile_vla_post_training/expert_concat/scripts/pi05_ExpertConcat_serve_policy.py).
#
#   ./scripts/pi05_ExpertConcat_infer.sh                        # dry run: the robot does not move
#   ./scripts/pi05_ExpertConcat_infer.sh user.dry_run=false     # execute on the robot
#   ./scripts/pi05_ExpertConcat_infer.sh user.policy_server=ws://127.0.0.1:8001 user.steps=20
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

exec python -m prometheus.workflows.rollout_sync \
  --config-name pi05_ExpertConcat_rollout_sync \
  "$@"
