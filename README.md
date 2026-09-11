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

Arm-only lifecycle (no cameras / tactile / messaging / recording):

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

FM target smoothing and motion limits are configured in `rollout_fm_sync.yaml`:

```yaml
user:
  smooth_action: true
  smooth_action_alpha: 0.6
  smooth_action_gripper: false
  joint_max_acceleration: 3.0  # rad/s^2; null restores velocity-only interpolation
```

EMA smooths policy targets across chunks before the scheduler. With acceleration
limits enabled, arm targets follow monotone cubic segments with continuous
velocity inside each chunk and zero endpoint velocity for the synchronous
inference pause. The trajectory is time-scaled to respect both velocity and
acceleration caps, so execution can take longer. It passes through all filtered
waypoints without stopping at each one or overshooting them. Grippers retain
linear interpolation and their velocity caps, sharing the arm trajectory timing.
These bounds describe the planned position target curve, not actual motor
acceleration or a hard real-time guarantee under transport stalls. The separate
SDK shutdown return still uses its existing `shutdown_safe_max_velocity` setting.

## V3 ZMQ transport

V3 uses local ZeroMQ IPC for robot commands/results, joint states, both EEF
poses, RealSense images/camera info, and Xense images/flow. Collection, replay,
sync inference, DAgger and run_env use the same transport. No ROS Python nodes,
DDS discovery, ROS message generation or cv_bridge are needed. The vendor A5
SDK still links native KDL parser libraries; keep those installed.

Run from the V3 directory in the existing `prometheusv3` environment:

```bash
cd /home/romoya-3/workspace/PrometheusV3
bash scripts/infer_fm.sh
```

Transport configuration (also available in the collection config):

```yaml
transport:
  backend: zmq
  namespace: ${oc.env:PROMETHEUS_ZMQ_NAMESPACE,prometheus-v3}
```

All cooperating processes must use the same namespace. IPC endpoints under
`/tmp/prometheus-zmq-<uid>/` are derived from namespace, logical domain and topic.
They are local to this machine and user. There is no broker to start. Duplicate
owners/publishers fail on an endpoint lock instead of replacing a live socket.
Subscribers reconnect when a publisher restarts.

The legacy `ros_domain_id` field is retained in runtime/stream metadata for
compatibility; its value now selects a ZMQ logical channel group. It does not
configure ROS. Control group 21 and sensor group 22 remain separate. The old
`RosDataBridge` import is an alias of `ZmqDataBridge`, not a ROS backend.

Images, poses and point clouds use typed MessagePack records with binary image
and float32 cloud payloads, without pickle on the wire. Live data uses bounded
PUB/SUB queues; overloaded receivers may lose frames. Commands use PUSH/PULL
with result acknowledgements, retries, request-id deduplication and expiration
before execution. A timeout cannot undo a command already accepted by the SDK.

CPU affinity remains active: A5 native control threads use CPUs 0–3; sync FM
inference, sensors and data processing use CPUs 8–15, configured before loading
the model. Joint speed limits, pedal controls, NumPy inputs and recording formats
are unchanged. Tactile difference video remains 350×200 MKV at the configured
0.5 scale; RGB remains 320×240 at its configured 0.5 scale.

The sensor-load diagnostic now compares `direct` with `zmq`; the jitter script
phases are named `shared-zmq-*` and `split-zmq-*`. Previous ROS experiment reports
are historical results, not measurements of this ZMQ implementation.

Offline validation (does not move real hardware):

```bash
python -m pytest -q tests
```

The ZMQ integration tests prohibit ROS imports and exercise separate processes,
robot commands/state/EEF, image/flow decoding and video recording, endpoint
ownership, subscriber reconnection, and the synchronous inference workflow with
a synthetic policy and hardware. Real FM inference and physical motion still
require a new on-device validation run.
