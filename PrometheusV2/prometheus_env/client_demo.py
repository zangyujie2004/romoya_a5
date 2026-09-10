from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from prometheus_env import connect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Prometheus ZMQ env client")
    parser.add_argument("--addr", default="tcp://127.0.0.1:5555")
    parser.add_argument("--max-steps", "--steps", dest="max_steps", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--obs-key", default="robot/qpos")
    parser.add_argument("--expect-action-space", default=None)
    parser.add_argument("--expect-action-dim", type=int, default=None)
    parser.add_argument("--expect-action-hz", type=float, default=None)
    parser.add_argument("--expect-policy-low-watermark", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=1)
    return parser.parse_args()


def build_expect(args: argparse.Namespace) -> dict[str, Any]:
    expect = {"action_chunk_horizon": int(args.chunk)}
    if args.expect_action_space is not None:
        expect["action_space"] = args.expect_action_space
    if args.expect_action_dim is not None:
        expect["action_dim"] = int(args.expect_action_dim)
    if args.expect_action_hz is not None:
        expect["action_hz"] = float(args.expect_action_hz)
    if args.expect_policy_low_watermark is not None:
        expect["policy_low_watermark"] = int(args.expect_policy_low_watermark)
    return expect


def load_policy(args: argparse.Namespace) -> Any:
    return None


def infer(policy: Any, obs: dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    if policy is not None:
        return np.asarray(policy.infer(obs), dtype=np.float32)
    qpos = np.asarray(obs[args.obs_key], dtype=np.float32).reshape(-1)
    return np.repeat(qpos[None, :], int(args.chunk), axis=0)


def main() -> None:
    args = parse_args()
    env = connect(args.addr)
    policy = load_policy(args)
    expect = build_expect(args)
    step_idx = 0
    start = time.monotonic()
    try:
        obs, info = env.reset(obs_keys=(args.obs_key,), expect=expect)
        print(f"[client_demo] reset ok available_obs={info.get('available_obs')}", flush=True)
        while int(args.max_steps) <= 0 or step_idx < int(args.max_steps):
            actions = infer(policy, obs, args)
            obs, reward, terminated, truncated, info = env.step(actions, obs_keys=(args.obs_key,))
            if int(args.print_every) > 0 and step_idx % int(args.print_every) == 0:
                print(
                    "[client_demo] step={} reward={:.3f} action_shape={} sent={}".format(
                        step_idx,
                        reward,
                        tuple(actions.shape),
                        info.get("sent_action_count"),
                    ),
                    flush=True,
                )
            step_idx += 1
            if terminated:
                print("[client_demo] server terminated episode; reset", flush=True)
                obs, _info = env.reset(obs_keys=(args.obs_key,), expect=expect)
            elif truncated:
                print("[client_demo] server truncated episode", flush=True)
                break
    finally:
        env.close()
    print(f"[client_demo] done steps={step_idx} duration_s={time.monotonic() - start:.2f}", flush=True)


if __name__ == "__main__":
    main()
