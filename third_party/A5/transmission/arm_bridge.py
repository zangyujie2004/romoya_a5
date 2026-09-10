from __future__ import annotations

import os
import sys

# 确保 arx .so 在 LD_LIBRARY_PATH 中，否则重启进程
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_lib  = os.path.join(_root, "bimanual", "lib", "arx_r5_src")
if _lib not in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(None, [_lib, os.path.dirname(_lib), os.environ.get("LD_LIBRARY_PATH", "")])
    )
    os.execv(sys.executable, [sys.executable] + sys.argv)
if _root not in sys.path:
    sys.path.insert(0, _root)

import json
import threading
import time
from typing import Any, Dict, Tuple

import numpy as np

try:
    import zmq
except ImportError:
    raise ImportError("请安装 pyzmq: pip install pyzmq")


# ─────────────────────────────── helpers ────────────────────────────────────

def _split7(raw) -> tuple[list[float], float]:
    """将底层 7 维数组拆为 (6 维关节, 夹爪标量)。"""
    a = np.asarray(raw, dtype=np.float64).ravel()
    joints  = a[:6].tolist() if len(a) >= 6 else ([0.0] * 6)
    gripper = float(a[6])    if len(a) >= 7 else 0.0
    return joints, gripper


# ─────────────────────────────── 状态读取 ────────────────────────────────────

def _arm_state(arm, prefix: str) -> Dict[str, Any]:
    """读取单臂状态:arm 为 None 时返回全零占位。"""
    if arm is None:
        return {
            f"{prefix}_joint_pos": [0.0] * 6, f"{prefix}_gripper_pos": 0.0,
            f"{prefix}_joint_vel": [0.0] * 6, f"{prefix}_gripper_vel": 0.0,
            f"{prefix}_joint_tor": [0.0] * 6, f"{prefix}_gripper_tor": 0.0,
            f"{prefix}_eef":       [0.0] * 6,
        }
    jpos, gpos = _split7(arm.get_joint_positions())
    jvel, gvel = _split7(arm.get_joint_velocities())
    jtor, gtor = _split7(arm.get_joint_currents())
    eef = np.asarray(arm.get_ee_pose_xyzrpy(), dtype=np.float64).tolist()
    return {
        f"{prefix}_joint_pos": jpos, f"{prefix}_gripper_pos": gpos,
        f"{prefix}_joint_vel": jvel, f"{prefix}_gripper_vel": gvel,
        f"{prefix}_joint_tor": jtor, f"{prefix}_gripper_tor": gtor,
        f"{prefix}_eef":       eef,
    }


def get_full_state(robot: Tuple[Any, Any]) -> Dict[str, Any]:
    """读取双臂完整状态，返回扁平字典。"""
    left_arm, right_arm = robot
    return {**_arm_state(left_arm, "left_arm"), **_arm_state(right_arm, "right_arm")}


# ─────────────────────────────── 命令分发 ────────────────────────────────────

def apply_cmd(robot: Tuple[Any, Any], cmd: Dict[str, Any]) -> None:
    """将命令字典分发到对应机械臂接口。"""
    if not cmd:
        return
    left_arm, right_arm = robot

    def _do(arm, gc_key, jpos_key, eef_key, grip_key):
        if arm is None:
            return
        if cmd.get(gc_key):
            arm.gravity_compensation()
        jpos = cmd.get(jpos_key)
        if isinstance(jpos, (list, tuple)) and len(jpos) >= 6:
            arm.set_joint_positions(np.array(jpos[:6], dtype=np.float64))
        eef = cmd.get(eef_key)
        if isinstance(eef, (list, tuple)) and len(eef) >= 6:
            arm.set_ee_pose_xyzrpy(xyzrpy=np.array(eef[:6], dtype=np.float64))
        grip = cmd.get(grip_key)
        if grip is not None:
            arm.set_gripper_pos(float(grip))

    _do(left_arm,  "left_arm_gc",  "left_arm_joint_pos",  "left_arm_eef",  "left_arm_gripper_pos")
    _do(right_arm, "right_arm_gc", "right_arm_joint_pos", "right_arm_eef", "right_arm_gripper_pos")


# ─────────────────────────────── ZMQ 服务 ────────────────────────────────────

def serve(
    left_can: str | None,
    right_can: str | None,
    state_pub_port: int = 5555,
    cmd_sub_port: int = 5556,
    state_hz: float = 100.0,
) -> None:
    """启动 ZMQ 服务left_can/right_can 为 None 则跳过该臂。"""
    from bimanual import SingleArm

    _arm = lambda can: SingleArm({"can_port": can, "type": 2}) if can else None
    robot = (_arm(left_can), _arm(right_can))
    print(f"[server] 左臂={'on:'+left_can if left_can else 'off'}  右臂={'on:'+right_can if right_can else 'off'}")

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    sub = ctx.socket(zmq.SUB)
    pub.bind(f"tcp://*:{state_pub_port}")
    sub.bind(f"tcp://*:{cmd_sub_port}")
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVTIMEO, 10)
    dt = 1.0 / state_hz

    print(f"[server] pub:{state_pub_port}  sub:{cmd_sub_port}  {state_hz:.0f}Hz")

    def _publish():
        while True:
            t0 = time.perf_counter()
            try:
                state = get_full_state(robot)
                lj = state["left_arm_joint_pos"]
                rj = state["right_arm_joint_pos"]
                print(f"[joint] L={[f'{v:.3f}' for v in lj]}  R={[f'{v:.3f}' for v in rj]}")
                pub.send(json.dumps(state).encode())
            except Exception as e:
                print(f"[pub] {e}", file=sys.stderr)
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))

    def _receive():
        while True:
            try:
                apply_cmd(robot, json.loads(sub.recv().decode()))
            except zmq.Again:
                pass
            except Exception as e:
                print(f"[sub] {e}", file=sys.stderr)

    threading.Thread(target=_publish, daemon=True).start()
    threading.Thread(target=_receive, daemon=True).start()

    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        ctx.destroy()
