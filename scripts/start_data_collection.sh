#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_NAME="${PROMETHEUS_CONFIG:-collect_a5}"

exec "${PYTHON_BIN}" -m prometheus.workflows.record_drag \
  --config-name "${CONFIG_NAME}" \
  "$@"
