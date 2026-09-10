#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export ROS_LOCALHOST_ONLY="${PROMETHEUS_ROS_LOCALHOST_ONLY:-1}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"

exec python -m prometheus.workflows.rollout_sync \
  --config-name rollout_fm_sync \
  "$@"

# Vision / tactile share this entry. Tactile streams are enabled automatically when
# checkpoint resolved_config has data.use_tactile=true (see policy/fm.yaml).
# Override run_dir: user.policy_run_dir=third_party/flow_matching/outputs/<task>
