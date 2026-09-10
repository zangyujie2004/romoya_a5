#!/usr/bin/env bash
set -euo pipefail

source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${source_dir}/build"

# Remove the vendor's high-frequency "ARX方舟无限" debug print while keeping
# every other controller diagnostic intact.
python "${source_dir}/patch_vendor_banner.py" \
    "${source_dir}/lib/arx_r5_src/libarx_r5a_src.so"

# --fresh discards copied CMake cache paths while preserving a repeatable build dir.
cmake --fresh -S "${source_dir}" -B "${build_dir}"
cmake --build "${build_dir}" --parallel
cmake --install "${build_dir}"
