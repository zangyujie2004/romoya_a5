# PrometheusV3 (A5)

## Environment

```bash
conda env create -f requirements/environment.yml
conda activate prometheusv3
python --version  # 3.12.x

python -m pip install -r requirements/requirements-cu128.txt
python -m pip install -r requirements/requirements.txt -e requirements
```

## A5 collection

Robot config: `prometheus/configs/robot/a5_drag.yaml` (follower `can1`/`can3`, 3 cameras, 4 Xense).

```bash
./prometheus/tools/restart_can1_can3.sh
./scripts/start_a5_drag_collection.sh
```

Arm-only lifecycle (no cameras / tactile / ROS / recording):

```bash
./scripts/test_a5_dual_arm_lifecycle.py
```

## A5 FM rollout

Defaults match collection: `robot=a5_drag`, `abs_qpos`, 30 Hz, no memory, horizon 128.

```bash
./scripts/infer_fm.sh
```

Replay a recorded joint episode:

```bash
./scripts/replay.sh user.input=/path/to/episode
```

ROS stays localhost-only unless you override:

```bash
PROMETHEUS_ROS_LOCALHOST_ONLY=0 bash scripts/start_a5_drag_collection.sh
```
