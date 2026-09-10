import os
import threading

import pytest

from prometheus.utils.cpu_affinity import inherit_cpu_affinity, normalize_cpu_affinity


def test_native_thread_inherits_mask_and_caller_is_restored():
    original = os.sched_getaffinity(0)
    selected = {min(original)}
    observed = []
    with inherit_cpu_affinity(list(selected)):
        worker = threading.Thread(target=lambda: observed.append(os.sched_getaffinity(0)))
        worker.start()
        worker.join()
        assert os.sched_getaffinity(0) == selected
    assert observed == [selected]
    assert os.sched_getaffinity(0) == original


def test_caller_affinity_is_restored_after_failure():
    original = os.sched_getaffinity(0)
    with pytest.raises(RuntimeError, match="startup failed"):
        with inherit_cpu_affinity([min(original)]):
            raise RuntimeError("startup failed")
    assert os.sched_getaffinity(0) == original


@pytest.mark.parametrize("value", [[], [-1], [True], [1.5]])
def test_invalid_cpu_affinity_fails(value):
    with pytest.raises(ValueError):
        normalize_cpu_affinity(value)
