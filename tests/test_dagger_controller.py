from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prometheus.dagger.controller import DAggerConfig, DAggerController
from prometheus.dagger.phase import DAggerPhase


@dataclass
class FakeRecorder:
    recording: bool = False
    paused: bool = False

    def pause(self) -> None:
        if not self.recording:
            raise RuntimeError("data recorder has no active episode to pause")
        self.paused = True

    def resume(self) -> None:
        if not self.recording:
            raise RuntimeError("data recorder has no active episode to resume")
        self.paused = False


@dataclass
class FakeDataSession:
    recorder: FakeRecorder = field(default_factory=FakeRecorder)
    episodes_started: list[tuple[str, dict[str, Any] | None]] = field(default_factory=list)
    episodes_stopped: list[bool] = field(default_factory=list)
    events: list[tuple[str, Any, int, dict[str, Any] | None]] = field(default_factory=list)

    def start_episode(self, episode_id: str, metadata: dict[str, Any] | None = None) -> Path:
        if self.recorder.recording:
            raise RuntimeError("data session already has an active episode")
        self.recorder.recording = True
        self.recorder.paused = False
        self.episodes_started.append((episode_id, metadata))
        return Path(episode_id)

    def stop_episode(self, *, accept: bool = True) -> None:
        self.recorder.recording = False
        self.recorder.paused = False
        self.episodes_stopped.append(bool(accept))

    def record_event(
        self,
        name: str,
        value: Any,
        *,
        stamp_ns: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        if stamp_ns is None:
            raise ValueError("stamp_ns is required in fake data session tests")
        self.events.append((name, value, int(stamp_ns), metadata))
        return True


@dataclass
class FakeRobotClient:
    reset_count: int = 0
    teach_count: int = 0
    position_count: int = 0
    reject_reset: bool = False

    def reset_home(self, *, timeout: float = 30.0, wait: bool = True) -> dict[str, Any]:
        self.reset_count += 1
        return {"accepted": not self.reject_reset, "timeout": timeout, "wait": wait}

    def set_teach_mode(self, *, timeout: float = 5.0, wait: bool = True) -> dict[str, Any]:
        self.teach_count += 1
        return {"accepted": True, "timeout": timeout, "wait": wait}

    def set_position_mode(self, *, timeout: float = 5.0, wait: bool = True) -> dict[str, Any]:
        self.position_count += 1
        return {"accepted": True, "timeout": timeout, "wait": wait}


def _make_controller(
    *,
    cycle_wait_s: float = 0.0,
    stamp_ns: int = 555_000_000,
    episode_index_start: int = 1,
) -> tuple[DAggerController, FakeRobotClient, FakeDataSession]:
    robot = FakeRobotClient()
    data = FakeDataSession()
    controller = DAggerController(
        robot_client=robot,
        data_session=data,
        config=DAggerConfig(cycle_wait_s=cycle_wait_s),
        run_id="run-test",
        get_robot_stamp_ns=lambda: stamp_ns,
        episode_index_start=episode_index_start,
        task_metadata={"task": "noop"},
    )
    return controller, robot, data


def _finish_cycle_wait(controller: DAggerController) -> None:
    assert controller.phase == DAggerPhase.CYCLE_WAIT
    assert controller.tick_cycle_wait() is True
    assert controller.phase == DAggerPhase.AUTONOMOUS


def _start_autonomous(controller: DAggerController) -> None:
    assert controller.apply_semantic("cycle").accepted is True
    _finish_cycle_wait(controller)


def test_initial_phase_is_idle():
    controller, _, _ = _make_controller()
    assert controller.phase == DAggerPhase.IDLE
    assert controller.can_infer is False
    assert controller.should_hold is False


def test_bootstrap_autonomous_starts_first_episode_without_reset_home():
    controller, robot, data = _make_controller()
    controller.bootstrap_autonomous()

    assert controller.phase == DAggerPhase.AUTONOMOUS
    assert controller.can_infer is True
    assert robot.reset_count == 0
    assert robot.position_count == 1
    assert data.episodes_started == [
        ("0001", {"workflow": "rollout_dagger", "run_id": "run-test", "task": "noop"})
    ]
    assert data.events == [("event_label", 0, 555_000_000, None)]


def test_bootstrap_then_cycle_uses_full_cycle_wait():
    controller, robot, data = _make_controller(cycle_wait_s=0.0)
    controller.bootstrap_autonomous()
    controller.apply_semantic("cycle")
    _finish_cycle_wait(controller)

    assert robot.reset_count == 1
    assert data.episodes_stopped == [True]
    assert data.episodes_started[-1][0] == "0002"
    assert controller.phase == DAggerPhase.AUTONOMOUS


def test_idle_cycle_enters_cycle_wait_without_starting_episode():
    controller, robot, data = _make_controller()
    result = controller.apply_semantic("cycle")

    assert result.accepted is True
    assert controller.phase == DAggerPhase.CYCLE_WAIT
    assert robot.reset_count == 1
    assert data.episodes_started == []
    assert data.episodes_stopped == []


def test_tick_cycle_wait_starts_episode_and_enters_autonomous():
    controller, robot, data = _make_controller()
    controller.apply_semantic("cycle")
    _finish_cycle_wait(controller)

    assert controller.phase == DAggerPhase.AUTONOMOUS
    assert data.episodes_started == [
        ("0001", {"workflow": "rollout_dagger", "run_id": "run-test", "task": "noop"})
    ]
    assert robot.position_count == 1
    assert data.recorder.recording is True
    assert data.recorder.paused is False
    assert controller.event_level == 0
    assert data.events == [("event_label", 0, 555_000_000, None)]


def test_autonomous_pause_is_deferred_until_enter_paused():
    controller, _, data = _make_controller()
    _start_autonomous(controller)

    hold = np.zeros(7, dtype=np.float32)
    result = controller.apply_semantic("pause")

    assert result.accepted is True
    assert result.deferred is True
    assert controller.pending_pause is True
    assert controller.phase == DAggerPhase.AUTONOMOUS
    assert data.recorder.paused is False

    controller.enter_paused(hold)

    assert controller.pending_pause is False
    assert controller.phase == DAggerPhase.PAUSED
    assert controller.should_hold is True
    assert data.recorder.paused is True
    assert np.array_equal(controller.hold_action, hold)


def test_autonomous_teaching_enters_correcting():
    controller, robot, data = _make_controller()
    _start_autonomous(controller)

    result = controller.apply_semantic("teaching")

    assert result.accepted is True
    assert controller.phase == DAggerPhase.CORRECTING
    assert robot.teach_count == 1
    assert data.recorder.paused is False
    assert controller.event_level == 1
    assert ("event_label", 1, 555_000_000, None) in data.events


def test_paused_teaching_enters_correcting():
    controller, robot, data = _make_controller()
    _start_autonomous(controller)
    controller.apply_semantic("pause")
    controller.enter_paused(np.zeros(3))

    result = controller.apply_semantic("teaching")

    assert result.accepted is True
    assert controller.phase == DAggerPhase.CORRECTING
    assert robot.teach_count == 1
    assert data.recorder.paused is False
    assert controller.event_level == 1


def test_correcting_teaching_is_ignored():
    controller, robot, _ = _make_controller()
    _start_autonomous(controller)
    controller.apply_semantic("teaching")

    result = controller.apply_semantic("teaching")

    assert result.ignored is True
    assert controller.phase == DAggerPhase.CORRECTING
    assert robot.teach_count == 1
    assert controller.ignored_events == 1


def test_cycle_wait_ignores_all_semantics():
    controller, robot, data = _make_controller()
    controller.apply_semantic("cycle")
    assert controller.phase == DAggerPhase.CYCLE_WAIT

    for semantic in ("cycle", "pause", "teaching"):
        result = controller.apply_semantic(semantic)
        assert result.ignored is True

    assert controller.phase == DAggerPhase.CYCLE_WAIT
    assert controller.ignored_events == 3
    assert robot.reset_count == 1
    assert data.episodes_started == []


def test_autonomous_cycle_stops_active_episode_before_wait():
    controller, robot, data = _make_controller()
    _start_autonomous(controller)
    data.events.clear()

    result = controller.apply_semantic("cycle")

    assert result.accepted is True
    assert controller.phase == DAggerPhase.CYCLE_WAIT
    assert data.episodes_stopped == [True]
    assert data.events == [("event_label", 0, 555_000_000, None)]
    assert data.recorder.recording is False
    assert robot.reset_count == 2


def test_correcting_cycle_stops_active_episode():
    controller, _, data = _make_controller()
    _start_autonomous(controller)
    controller.apply_semantic("teaching")
    data.events.clear()

    result = controller.apply_semantic("cycle")

    assert result.accepted is True
    assert controller.phase == DAggerPhase.CYCLE_WAIT
    assert data.episodes_stopped == [True]
    assert data.events == [("event_label", 0, 555_000_000, None)]


def test_episode_index_increments_across_cycles():
    controller, _, data = _make_controller()
    _start_autonomous(controller)
    controller.apply_semantic("cycle")
    _finish_cycle_wait(controller)

    assert data.episodes_started[-1][0] == "0002"


def test_record_event_uses_robot_stamp():
    controller, _, data = _make_controller(stamp_ns=777_000_000)
    _start_autonomous(controller)
    controller.apply_semantic("teaching")

    assert data.events[-1] == ("event_label", 1, 777_000_000, None)


def test_paused_and_correcting_ignore_pause():
    controller, _, _ = _make_controller()
    _start_autonomous(controller)
    controller.apply_semantic("pause")
    controller.enter_paused(np.zeros(2))

    paused_result = controller.apply_semantic("pause")
    assert paused_result.ignored is True

    controller.apply_semantic("teaching")
    correcting_result = controller.apply_semantic("pause")
    assert correcting_result.ignored is True


def test_enter_paused_requires_pending_pause():
    controller, _, _ = _make_controller()
    _start_autonomous(controller)

    with pytest.raises(RuntimeError, match="pending_pause"):
        controller.enter_paused(np.zeros(2))


def test_reset_home_rejection_raises():
    controller, robot, _ = _make_controller()
    robot.reject_reset = True

    with pytest.raises(RuntimeError, match="reset_home rejected"):
        controller.apply_semantic("cycle")


def test_can_infer_only_in_autonomous_without_pending_pause():
    controller, _, _ = _make_controller()
    assert controller.can_infer is False

    _start_autonomous(controller)
    assert controller.can_infer is True

    controller.apply_semantic("pause")
    assert controller.can_infer is False

    controller.enter_paused(np.zeros(2))
    assert controller.can_infer is False

    controller.apply_semantic("teaching")
    assert controller.can_infer is False


def test_apply_event_alias_matches_apply_semantic():
    assert DAggerController.apply_event is DAggerController.apply_semantic
