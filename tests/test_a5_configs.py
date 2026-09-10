from __future__ import annotations

from pathlib import Path
import struct

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from prometheus.sessions.event import EventSession, PedalInput

from prometheus.utils.workflow import RunPaths, build_hardware_session


CONFIG_DIR = Path(__file__).resolve().parents[1] / "prometheus" / "configs"


def test_collect_a5_uses_slave_arms_and_three_cameras(tmp_path):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="collect_a5")
    data = OmegaConf.to_container(cfg, resolve=True)

    assert data["robot"]["id"] == "a5_drag"
    assert data["robot"]["robot"]["can"] == {"left": "can1", "right": "can3"}
    assert data["robot"]["cameras"]["ir"] is False
    assert data["robot"]["cameras"]["prepub_resize"] == 0.5
    assert data["workflow"]["reset_after_episode"] == {
        "enabled": True,
        "timeout_s": 30.0,
        "on_shutdown": False,
    }
    assert data["workflow"]["shutdown_safe_pose"] == {
        "enabled": True,
        "timeout_s": 65.0,
    }
    assert data["workflow"]["preview"]["show_tactile"] is True
    assert data["workflow"]["preview"]["tactile_noise_floor_m"] == 0.0005
    assert data["workflow"]["preview"]["tactile_max_deformation_m"] == 0.01
    assert data["workflow"]["preview"]["tactile_median_kernel"] == 3
    assert data["robot"]["tactile"]["publish_flow"] is True
    assert data["robot"]["tactile"]["prepub_resize"] == 0.5
    assert data["robot"]["tactile"]["publish_difference"] is True
    assert data["robot"]["tactile"]["publish_raw"] is False
    assert data["robot"]["tactile"]["serial_no"] == {
        "left_wrist_0": "OG001596",
        "left_wrist_1": "OG001595",
        "right_wrist_0": "OG001593",
        "right_wrist_1": "OG001594",
    }

    session = _session(data, tmp_path)

    assert session.robot["driver"]["_target_"] == "prometheus.nodes.robots.a5.A5DragDriver"
    assert session.robot["driver"]["left_can"] == "can1"
    assert session.robot["driver"]["right_can"] == "can3"
    assert session.robot["driver"]["link6_mass_kg"] == data["robot"]["robot"]["link6_mass_kg"]
    assert session.robot["driver"]["teach_gripper_torque"] == {"left": 0.0, "right": 0.0}
    assert not any(key.startswith("gripper_open") for key in session.robot["driver"])
    assert session.robot["driver"]["home_timeout_s"] == 25.0
    assert session.robot["driver"]["shutdown_safe_position"] == {
        "left": [-0.00767836, 0.00171661, -0.01231598, -0.16930453, -0.00083383, 0.03261662],
        "right": [0.00424513, 0.00362396, -0.01498627, -0.06973903, -0.00922616, -0.01735687],
    }
    assert session.robot["driver"]["shutdown_safe_max_velocity"] == 0.10
    assert session.robot["driver"]["shutdown_safe_timeout_s"] == 60.0
    assert session.robot["driver"]["shutdown_safe_position_tolerance"] == 0.03
    assert set(session.sensors) == {"realsense", "tactile"}
    assert session.sensors["realsense"]["cameras"] == {
        "base_0": "262622275297",
        "left_wrist_0": "260322274232",
        "right_wrist_0": "260422275259",
    }
    assert session.sensors["realsense"]["enable_infra"] is False
    assert session.sensors["realsense"]["resize"] == 0.5
    assert session.sensors["tactile"]["sensors"] == {
        "left_wrist_0": "OG001596",
        "left_wrist_1": "OG001595",
        "right_wrist_0": "OG001593",
        "right_wrist_1": "OG001594",
    }
    assert session.sensors["tactile"]["require_gpu"] is True
    assert session.sensors["tactile"]["inference_threads"] == 1
    assert session.sensors["tactile"]["prepub_resize"] == 0.5
    assert session.sensors["tactile"]["publish_difference"] is True
    assert session.sensors["tactile"]["publish_raw"] is False
    assert session.robot["driver"]["control_cpu_affinity"] == [0, 1, 2, 3]
    assert data["workflow"]["data_cpu_affinity"] == list(range(8, 16))
    assert "base_0_color" in session.streams
    assert not any("infra1" in name or "infra2" in name for name in session.streams)
    assert "left_wrist_0_tactile_flow" in session.streams
    assert "right_wrist_1_tactile_flow" in session.streams
    assert {
        "left_wrist_0_tactile_difference",
        "left_wrist_1_tactile_difference",
        "right_wrist_0_tactile_difference",
        "right_wrist_1_tactile_difference",
    }.issubset(session.streams)


def test_rollout_fm_uses_a5_collect_robot(tmp_path):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="rollout_fm_sync")
    data = OmegaConf.to_container(cfg, resolve=True)

    assert data["robot"]["id"] == "a5_drag"
    assert data["robot"]["robot"]["action_space"] == "abs_qpos"
    assert data["robot"]["robot"]["state_publish_hz"] == 100
    assert data["robot"]["robot"]["can"] == {"left": "can1", "right": "can3"}
    assert data["user"]["n_action_steps"] == 128
    assert data["user"]["use_memory"] is False
    assert data["user"]["action_hz"] == 30.0
    assert data["run"]["wait_ready_timeout_s"] == 15.0
    assert data["data"]["bridge"]["stamp_source"] == "header"
    assert data["robot"]["cameras"]["ir"] is False
    assert data["robot"]["cameras"]["prepub_resize"] == 0.5
    assert data["robot"]["cameras"]["serial_no"] == {
        "base_0": "262622275297",
        "left_wrist_0": "260322274232",
        "right_wrist_0": "260422275259",
    }
    assert data["robot"]["tactile"]["serial_no"] == {
        "left_wrist_0": "OG001596",
        "left_wrist_1": "OG001595",
        "right_wrist_0": "OG001593",
        "right_wrist_1": "OG001594",
    }

    session = _session(data, tmp_path)
    assert session.robot["driver"]["_target_"] == "prometheus.nodes.robots.a5.A5DragDriver"
    assert session.robot["owner"]["hz"] == 100
    assert session.robot["driver"]["hz"] == 100
    assert session.sensors["realsense"]["enable_infra"] is False
    assert "base_0_color" in session.streams
    assert "left_wrist_0_tactile_flow" in session.streams
    assert not any("infra1" in name or "infra2" in name for name in session.streams)


def test_hardware_session_does_not_import_specific_hardware_types():
    source = (Path(__file__).resolve().parents[1] / "prometheus" / "sessions" / "hardware.py").read_text()

    assert "arx5" not in source
    assert "a5" not in source
    assert "realsense" not in source
    assert "expand_robot_runtime" not in source


def test_a5_four_key_pedal_requires_two_released_presses(monkeypatch):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="collect_a5")
    event_cfg = OmegaConf.to_container(cfg.event, resolve=True)
    assert event_cfg["tty"]["confirm_presses"] == 2
    assert event_cfg["pedal"]["confirm_presses"] == 2
    assert event_cfg["pedal"]["pedal_map"] == {2: 0, 3: 1}

    events = EventSession()
    events.start()
    try:
        pedal = PedalInput(events, **event_cfg["pedal"])
        evdev = struct.Struct("llHHi")

        def send(code, value, now):
            monkeypatch.setattr("prometheus.sessions.event.time.monotonic", lambda: now)
            pedal._process_evdev_bytes(evdev.pack(0, 0, 1, code, value))

        for code, event_id in ((2, 0), (3, 1)):
            # Neither auto-repeat nor duplicate down events count as a second press.
            send(code, 1, 0.0)
            assert events.poll() is None
            send(code, 2, 0.1)
            send(code, 1, 0.2)
            assert events.poll() is None
            send(code, 0, 0.3)
            send(code, 1, 0.5)
            assert events.poll() == (event_id, True)
            send(code, 2, 0.6)
            assert events.poll() is None
            send(code, 0, 0.7)

            # A late second press starts a fresh pair, rather than triggering.
            send(code, 1, 2.0)
            send(code, 0, 2.1)
            send(code, 1, 3.2)
            assert events.poll() is None
            send(code, 0, 3.3)
            send(code, 1, 3.5)
            assert events.poll() == (event_id, True)
            send(code, 0, 3.6)

        # The observed third/fourth-key codes and old a/b/c codes are ignored.
        for code in (4, 5, 30, 48, 46):
            pedal._process_evdev_bytes(evdev.pack(0, 0, 1, code, 1))
            pedal._process_evdev_bytes(evdev.pack(0, 0, 1, code, 0))
        assert events.poll() is None
    finally:
        events.close()


def _session(data, tmp_path):
    run = RunPaths(
        run_id="test",
        output_dir=tmp_path,
        runtime_dir=tmp_path / "runtime",
        manifest_path=tmp_path / "workflow_manifest.json",
    )
    return build_hardware_session(
        run=run,
        hardware_cfg=data["hardware"],
        robot_cfg=data["robot"],
    )
