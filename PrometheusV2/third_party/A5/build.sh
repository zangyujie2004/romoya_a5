#!/usr/bin/env bash
set -euo pipefail

sdk_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${sdk_root}/bimanual/build.sh"
