#!/usr/bin/env python3
"""Final ProPainter process and status helpers for Remove_Hand."""

from __future__ import annotations

import json
import os
import resource
import subprocess
import time
from pathlib import Path

def raise_nofile_limit(minimum: int=65536) -> None:
    """Raise the per-process file descriptor limit before ProPainter."""
    try:
        import resource
        (soft, hard) = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = minimum if hard == resource.RLIM_INFINITY else min(minimum, hard)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        (current, _) = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f'[green-v4] nofile soft={current} hard={hard}', flush=True)
    except (ImportError, OSError, ValueError) as exc:
        print(f'[green-v4] warning: unable to raise nofile limit: {exc}', flush=True)

def write_status(out: Path, stage: str, **extra: object) -> None:
    payload = {'stage': stage, 'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    payload.update(extra)
    temporary = out / 'status.json.tmp'
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(out / 'status.json')

def run_checked(command: list[str], cwd: Path | None=None) -> None:
    print('[green-v3] exec:', ' '.join(command), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)

def find_propainter_result(root: Path, clip_stem: str) -> Path:
    expected = root / clip_stem / 'inpaint_out.mp4'
    if expected.is_file():
        return expected
    matches = list(root.glob('*/inpaint_out.mp4'))
    if len(matches) != 1:
        raise RuntimeError(f'unable to find unique ProPainter result under {root}')
    return matches[0]
