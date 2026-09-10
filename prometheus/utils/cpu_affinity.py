from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any


def normalize_cpu_affinity(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    cpus = tuple(value)
    if not cpus or any(type(cpu) is not int or cpu < 0 for cpu in cpus):
        raise ValueError("cpu_affinity must be a non-empty list of nonnegative CPU ids")
    return tuple(sorted(set(cpus)))


def _set_affinity(tid: int, cpus: tuple[int, ...]) -> None:
    os.sched_setaffinity(tid, cpus)
    if set(os.sched_getaffinity(tid)) != set(cpus):
        raise RuntimeError(f"CPU affinity {cpus} was not fully applied to thread {tid}")


def configure_process_cpu_affinity(value: Any) -> None:
    """Apply the data workload CPU set before hardware threads are created."""
    cpus = normalize_cpu_affinity(value)
    if cpus is None:
        return
    for task in Path("/proc/self/task").iterdir():
        try:
            _set_affinity(int(task.name), cpus)
        except ProcessLookupError:
            pass


@contextmanager
def inherit_cpu_affinity(value: Any):
    """New native threads inherit this caller's mask; restore the caller only."""
    cpus = normalize_cpu_affinity(value)
    if cpus is None:
        yield
        return
    previous = os.sched_getaffinity(0)
    try:
        _set_affinity(0, cpus)
        yield
    finally:
        os.sched_setaffinity(0, previous)
