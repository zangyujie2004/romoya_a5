#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# A5 joint replay uses native SDK gripper positions from the recorded trajectory.
# dry_run only skips trajectory commands: hardware startup and reset_home still run.
exec "${PYTHON_BIN:-python}" -m prometheus.workflows.replay \
  robot=a5_drag \
  user.input=/home/romoya-3/workspace/raw_data/egg_0908_100/0000 \
  user.control_mode=joint \
  user.dry_run=false \
  "$@"
