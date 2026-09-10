#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


# ControllerThread::ArmThread contains an unconditional branch to a std::cerr
# debug print every fifth control loop. Replacing that branch with NOPs preserves
# the control-loop fallthrough and all other vendor diagnostics.
BEFORE = bytes.fromhex("3dcccccc0c0f861b0100004183c501")
AFTER = bytes.fromhex("3dcccccc0c9090909090904183c501")
BANNER = "ARX方舟无限".encode()


def patch_library(path: Path) -> str:
    data = path.read_bytes()
    if BANNER not in data:
        raise RuntimeError(f"vendor banner not found in {path}")
    before_count = data.count(BEFORE)
    after_count = data.count(AFTER)
    if before_count == 1 and after_count == 0:
        path.write_bytes(data.replace(BEFORE, AFTER, 1))
        return "patched"
    if before_count == 0 and after_count == 1:
        return "already patched"
    raise RuntimeError(
        f"unexpected A5 vendor library layout in {path}: "
        f"before={before_count}, after={after_count}"
    )


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} LIBARX_R5A_SRC_SO")
    path = Path(sys.argv[1]).resolve()
    print(f"{path}: {patch_library(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
