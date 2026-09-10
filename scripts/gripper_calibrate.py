from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate one X5 gripper or measure one A5 gripper's mechanical endpoints."
    )
    parser.add_argument("--model", default="X5", help="X5 (SDK calibration) or A5 (endpoint measurement)")
    parser.add_argument("--interface", default="can1")
    parser.add_argument("--side", choices=("left", "right"), help="A5 config side; inferred for can1/can3")
    parser.add_argument(
        "--init_open_readout",
        type=float,
        default=100.0,
        help=(
            "Temporary permissive value used only to let Arx5JointController initialize. "
            "Use the printed fully-open readout after calibration as the real config value."
        ),
    )
    parser.add_argument("--width", type=float, default=0.082)
    parser.add_argument("--no_background", action="store_true")
    parser.add_argument("--sdk_root", default=str(PROJECT_ROOT / "third_party" / "A5"))
    parser.add_argument("--samples", type=int, default=25, help="A5 samples per endpoint (default: 25)")
    parser.add_argument(
        "--sample_interval_s",
        type=float,
        default=0.01,
        help="A5 interval between endpoint samples (default: 0.01)",
    )
    return parser.parse_args()


def calibrate_arx5(args: argparse.Namespace) -> None:
    import arx5_interface as arx5

    print(
        f"Preparing {args.model} gripper calibration on {args.interface}.\n"
        f"Temporary controller init config: gripper_open_readout={args.init_open_readout}, "
        f"gripper_width={args.width}.\n"
        "This temporary value is not the final calibration value.\n"
        "Stop all other ARX processes on this CAN interface before continuing.",
        flush=True,
    )
    input("Press Enter only when the robot is safe and the CAN interface is idle...")

    robot_cfg = arx5.RobotConfigFactory.get_instance().get_config(args.model)
    robot_cfg.gripper_open_readout = float(args.init_open_readout)
    robot_cfg.gripper_width = float(args.width)

    ctrl_cfg = arx5.ControllerConfigFactory.get_instance().get_config("joint_controller", robot_cfg.joint_dof)
    ctrl_cfg.background_send_recv = not args.no_background

    controller = arx5.Arx5JointController(robot_cfg, ctrl_cfg, args.interface)
    print(
        "Controller initialized. Now follow the SDK prompts exactly:\n"
        "1. fully close the gripper, then press Enter;\n"
        "2. fully open the gripper, then press Enter.\n"
        "After it prints 'Fully-open joint position readout', use that value as the real "
        "gripper_open_readout for this CAN arm.",
        flush=True,
    )
    controller.calibrate_gripper()


def sample_a5_gripper(arm: object, *, samples: int, interval_s: float) -> tuple[float, float]:
    import numpy as np

    if samples <= 0:
        raise ValueError("--samples must be positive")
    if interval_s < 0.0:
        raise ValueError("--sample_interval_s must be non-negative")
    values = []
    for index in range(samples):
        joints = np.asarray(arm.get_joint_positions(), dtype=np.float64).reshape(-1)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise RuntimeError(f"Invalid A5 joint feedback while sampling: {joints}")
        values.append(float(joints[6]))
        if index + 1 < samples and interval_s > 0.0:
            time.sleep(interval_s)
    return float(np.median(values)), float(np.ptp(values))


def calibrate_a5(args: argparse.Namespace) -> None:
    from prometheus.nodes.robots.a5 import A5DragDriver

    side = args.side or {"can1": "left", "can3": "right"}.get(args.interface)
    side_label = side or "SIDE"
    print(
        f"Preparing A5 gripper endpoint measurement on {args.interface}.\n"
        "This reads the mechanical endpoints; it does not modify motor firmware.\n"
        "Stop the collection system and every other process using this CAN interface.\n"
        "The SDK will request gravity compensation and set extra gripper torque to zero.",
        flush=True,
    )
    input("Press Enter only when the arm is supported and the CAN interface is idle...")

    loader = A5DragDriver(sdk_root=args.sdk_root)
    single_arm = loader._import_single_arm()
    arm = None
    try:
        arm = single_arm(
            {
                "can_port": args.interface,
                "urdf_name": "a5.urdf",
                "num_joints": 7,
                "dt": 0.01,
            }
        )
        arm.set_gripper_ext_torque(0.0)
        arm.gravity_compensation()

        input("Move the gripper by hand to its fully CLOSED mechanical endpoint, then press Enter...")
        closed, closed_spread = sample_a5_gripper(
            arm,
            samples=args.samples,
            interval_s=args.sample_interval_s,
        )
        print(f"Fully-closed readout: {closed:.6f} (sample spread {closed_spread:.6f})", flush=True)

        input("Move the gripper by hand to its fully OPEN mechanical endpoint, then press Enter...")
        opened, open_spread = sample_a5_gripper(
            arm,
            samples=args.samples,
            interval_s=args.sample_interval_s,
        )
        print(f"Fully-open readout:   {opened:.6f} (sample spread {open_spread:.6f})", flush=True)
        if opened == closed:
            raise RuntimeError(
                f"A5 {side_label} gripper returned the same endpoint readout "
                f"({closed:.6f}) on {args.interface}. If the gripper was physically moved "
                "between closed and open, check the gripper motor's power, CAN feedback "
                "and configured motor ID. No valid endpoints were measured."
            )
        if opened >= closed:
            raise RuntimeError(
                "Unexpected A5 endpoints: the fully-open readout should be more negative than "
                "the fully-closed readout; check the order and repeat"
            )

        print(f"\n{side_label} A5 gripper endpoints: closed={closed:.6f}, open={opened:.6f}", flush=True)
    finally:
        if arm is not None:
            try:
                arm.set_gripper_ext_torque(0.0)
                arm.protect_mode()
            finally:
                del arm
                gc.collect()


def main() -> None:
    args = parse_args()
    if not sys.stdin.isatty():
        raise RuntimeError("Run this script from an interactive terminal; calibration must wait for Enter.")
    if str(args.model).strip().upper() == "A5":
        calibrate_a5(args)
    else:
        calibrate_arx5(args)


if __name__ == "__main__":
    main()
