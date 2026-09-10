#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

for interface in can1 can3; do
    state_file="/sys/class/net/${interface}/operstate"
    if [[ ! -r "${state_file}" ]] || [[ "$(<"${state_file}")" != "up" ]]; then
        echo "${interface} is not UP. Run prometheus/tools/restart_can1_can3.sh first." >&2
        exit 1
    fi
done

export PROMETHEUS_CONFIG=collect_a5
exec "${PROJECT_ROOT}/scripts/start_data_collection.sh" "$@"
