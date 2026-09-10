#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_PYTHON = Path.home() / "miniconda3" / "envs" / "prometheusv3" / "bin" / "python"
BOOTSTRAP_ENV = "PROMETHEUS_A5_LIFECYCLE_BOOTSTRAPPED"


class TerminationRequested(Exception):
    def __init__(self, signum: int):
        super().__init__(f"received signal {signum}")
        self.signum = int(signum)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test only the A5 follower-arm lifecycle: gravity compensation while running, "
            "home on Ctrl-C, then protect and exit."
        )
    )
    parser.add_argument(
        "--no-auto-can",
        action="store_true",
        help="fail instead of starting can1/can3 when either interface is down",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify the robot environment and A5 config without opening CAN or connecting arms",
    )
    parser.add_argument(
        "--poll-hz",
        type=float,
        default=50.0,
        help="joint-state polling frequency while waiting for Ctrl-C (default: 50)",
    )
    parser.add_argument(
        "--countdown-s",
        type=int,
        default=3,
        help="safety countdown before connecting the arms (default: 3)",
    )
    return parser.parse_args()


def ensure_robot_environment() -> None:
    if not ROBOT_PYTHON.is_file():
        raise FileNotFoundError(f"prometheusv3 Conda Python not found: {ROBOT_PYTHON}")

    env = os.environ.copy()
    library_dirs = (
        PROJECT_ROOT / "third_party" / "A5" / "bimanual" / "api" / "arx_r5_src",
        PROJECT_ROOT / "third_party" / "A5" / "bimanual" / "api",
        Path("/usr/local/lib"),
    )
    current_ld_path = [item for item in env.get("LD_LIBRARY_PATH", "").split(":") if item]
    env["LD_LIBRARY_PATH"] = ":".join(
        [str(path) for path in library_dirs]
        + [item for item in current_ld_path if item not in {str(path) for path in library_dirs}]
    )
    env["CONDA_DEFAULT_ENV"] = "prometheusv3"
    env["CONDA_PREFIX"] = str(ROBOT_PYTHON.parents[1])
    current_python_path = [item for item in env.get("PYTHONPATH", "").split(":") if item]
    env["PYTHONPATH"] = ":".join(
        [str(PROJECT_ROOT)]
        + [item for item in current_python_path if item != str(PROJECT_ROOT)]
    )

    running_robot_python = Path(sys.executable).resolve() == ROBOT_PYTHON.resolve()
    if not running_robot_python or env.get(BOOTSTRAP_ENV) != "1":
        env[BOOTSTRAP_ENV] = "1"
        os.execve(str(ROBOT_PYTHON), [str(ROBOT_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]], env)


def can_is_up(interface: str) -> bool:
    state_path = Path("/sys/class/net") / interface / "operstate"
    return state_path.is_file() and state_path.read_text(encoding="ascii").strip() == "up"


def ensure_can_interfaces(*, auto_start: bool) -> None:
    interfaces = ("can1", "can3")
    down = [interface for interface in interfaces if not can_is_up(interface)]
    if down and auto_start:
        restart_script = PROJECT_ROOT / "prometheus" / "tools" / "restart_can1_can3.sh"
        print(f"[can] starting can1/can3 with {restart_script}", flush=True)
        subprocess.run([str(restart_script)], cwd=PROJECT_ROOT, check=True)
        down = [interface for interface in interfaces if not can_is_up(interface)]
    if down:
        raise RuntimeError(
            f"CAN interfaces are not UP: {down}. Run prometheus/tools/restart_can1_can3.sh first."
        )


def load_driver() -> Any:
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from omegaconf import OmegaConf

    from prometheus.nodes.robots.a5 import A5DragDriver, runtime_config

    config_path = PROJECT_ROOT / "prometheus" / "configs" / "robot" / "a5_drag.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    driver_config = runtime_config(config)["driver"]
    kwargs = {key: value for key, value in driver_config.items() if key != "_target_"}
    return A5DragDriver(**kwargs)


def request_termination(signum: int, _frame: Any) -> None:
    raise TerminationRequested(signum)


def print_state(state: Any) -> None:
    left = state.qpos[:7].round(3).tolist()
    right = state.qpos[7:].round(3).tolist()
    print(f"[state] left={left} right={right}", flush=True)


def run_lifecycle(driver: Any, *, poll_hz: float, countdown_s: int) -> int:
    if poll_hz <= 0:
        raise ValueError("--poll-hz must be positive")
    if countdown_s < 0:
        raise ValueError("--countdown-s must be non-negative")

    print("A5 dual-arm lifecycle test", flush=True)
    print("  left follower : can1", flush=True)
    print("  right follower: can3", flush=True)
    status = driver.status()
    print(f"  link6 mass kg : {status['link6_mass_kg']}", flush=True)
    print(f"  gripper torque: {status['teach_gripper_torque']}", flush=True)
    print("  Ctrl-C        : home both arms, wait, then protect and exit", flush=True)
    print("  second Ctrl-C : cancel homing and protect immediately", flush=True)
    print("Clear the workspace and keep the emergency stop within reach.", flush=True)
    for remaining in range(countdown_s, 0, -1):
        print(f"Connecting in {remaining}...", flush=True)
        time.sleep(1.0)

    ready = False
    exit_code = 0
    try:
        print("[robot] connecting can1/can3...", flush=True)
        driver.connect()
        ready = True
        print("[robot] gravity compensation active; drag the arms or press Ctrl-C to test reset.", flush=True)

        period = 1.0 / poll_hz
        next_poll = time.monotonic()
        next_report = next_poll
        while True:
            state = driver.read_state()
            now = time.monotonic()
            if now >= next_report:
                print_state(state)
                next_report = now + 1.0
            next_poll += period
            time.sleep(max(0.0, next_poll - time.monotonic()))
    except KeyboardInterrupt:
        if not ready:
            print("\n[robot] Ctrl-C before startup completed; skipping home.", flush=True)
            return 130
        print("\n[robot] Ctrl-C received; homing both arms now.", flush=True)
        print("[robot] Press Ctrl-C again to cancel homing and enter protect mode.", flush=True)
        try:
            driver.reset_home()
        except KeyboardInterrupt:
            print("\n[robot] homing cancelled by second Ctrl-C.", flush=True)
            exit_code = 130
        except Exception:
            print("[robot] home reset failed; protect mode has been requested.", file=sys.stderr, flush=True)
            traceback.print_exc()
            exit_code = 1
        else:
            print("[robot] home reached and gravity compensation restored.", flush=True)
    except TerminationRequested as exc:
        print(f"\n[robot] signal {exc.signum} received; skipping home.", flush=True)
        exit_code = 128 + exc.signum
    except Exception:
        print("[robot] unexpected error; skipping home and entering protect mode.", file=sys.stderr, flush=True)
        traceback.print_exc()
        exit_code = 1
    finally:
        print("[robot] entering protect mode and closing.", flush=True)
        try:
            driver.close()
        except Exception:
            traceback.print_exc()
            exit_code = 1
        print("[robot] lifecycle test finished.", flush=True)
    return exit_code


def main() -> int:
    args = parse_args()
    ensure_robot_environment()
    os.chdir(PROJECT_ROOT)
    signal.signal(signal.SIGTERM, request_termination)
    signal.signal(signal.SIGHUP, request_termination)
    driver = load_driver()
    if args.check_only:
        status = driver.status()
        print(f"robot_python={sys.executable}")
        print(f"left_can={status['left_can']} right_can={status['right_can']}")
        print(f"sdk_root={status['sdk_root']}")
        print(f"link6_mass_kg={status['link6_mass_kg']}")
        print(f"teach_gripper_torque={status['teach_gripper_torque']}")
        print("A5 lifecycle environment check passed; no hardware was opened.")
        return 0
    ensure_can_interfaces(auto_start=not args.no_auto_can)
    return run_lifecycle(driver, poll_hz=float(args.poll_hz), countdown_s=int(args.countdown_s))


if __name__ == "__main__":
    raise SystemExit(main())
