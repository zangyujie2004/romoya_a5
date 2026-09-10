# PrometheusV3

## Environment

Create the local development environment:

```bash
conda env create -f environment.yml
conda activate prometheusv3
python --version  # 3.12.x

# RTX 5090: install the official CUDA 12.8 wheels before other pip dependencies.
python -m pip install -r requirements-cu128.txt
python -m pip install -r requirements.txt -e .
```

The environment includes ROS 2 Jazzy for Python 3.12. See
[Ubuntu 24.04 environment notes](docs/ENVIRONMENT_UBUNTU24.md) for the local
RealSense binding and hardware SDK setup.

## ROS Runtime Defaults

PrometheusV3 defaults ROS 2 communication to local-only mode for data collection.
This avoids receiving DDS traffic from other machines on the same network and
prevents large image messages from being corrupted by non-local transport.

The collection config sets:

```yaml
ros:
  localhost_only: true
```

`scripts/start_data_collection.sh` also exports the matching ROS environment by
default:

```bash
export ROS_LOCALHOST_ONLY=1
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
```

Use the normal collection entry point:

```bash
bash scripts/start_data_collection.sh
```

## A5 Drag Collection

For local startup commands and validation results, see
[A5 collection test](docs/A5_COLLECTION.md).

The A5 drag configuration uses only the follower arms:

- left follower: `can1`
- right follower: `can3`
- recorded joint order: left 7 values, then right 7 values
- tactile sensors: four Xense sensors with GPU inference
- normal episode end: home both arms, wait for settling, then return to gravity compensation
- shutdown, interruption, or reset failure: enter protect mode without automatic homing

After activating the environment and powering the follower arms, start the CAN
interfaces and collection workflow with:

```bash
./prometheus/tools/restart_can1_can3.sh
./scripts/start_a5_drag_collection.sh
```

To test only the two follower arms without cameras, tactile sensors, ROS, or
recording, run:

```bash
./scripts/test_a5_dual_arm_lifecycle.py
```

The script enters gravity compensation after connecting. Its first `Ctrl-C`
homes both arms and then protects them before exit. A second `Ctrl-C` during
homing cancels the reset and requests protect mode immediately.

The current three-camera configuration is in
`prometheus/configs/robot/a5_drag.yaml`. The base camera serial has been
visually confirmed; verify the two wrist labels by covering one wrist camera
before the first real recording.

Only override this when you intentionally need ROS 2 traffic across machines:

```bash
PROMETHEUS_ROS_LOCALHOST_ONLY=0 bash scripts/start_data_collection.sh
```
