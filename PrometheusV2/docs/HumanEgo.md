# HumanEgo Deployment

HumanEgo is integrated as a remote policy. PrometheusV3 owns workflow
orchestration, hardware lifecycle, data alignment, and action execution.
HumanEgo owns visual preprocessing, depth/object initialization, ICT inference,
and action decoding behind a websocket server.

Runtime path:

```text
rollout_sync workflow
  -> DataSession.frame()
  -> HumanEgoPolicy.infer()
  -> RemoteInferenceClient
  -> third_party/HumanEgo/inference/serve_policy.py
  -> ActionChunk(action_space="abs_eef")
  -> ActionScheduler
```

The robot-side policy adapter lives in:

```text
prometheus/policy/humanego_policy.py
prometheus/configs/policy/humanego.yaml
prometheus/configs/rollout_humanego_sync.yaml
```

The GPU-side services live in:

```text
third_party/HumanEgo/
third_party/fast-foundationstereo/
```

The Prometheus-specific server entry points are maintained inside those
submodules.

## Data Contract

First inference sends:

```text
rgb_bgr, K
stereo[left, right, K, baseline_m]
eef_in_base[left, right]
gripper_widths[left, right]
gripper_width_m
```

Later inference sends the same payload without `stereo`. `reset_episode=true`
is sent only on the first request after constructing `HumanEgoPolicy`.

The policy adapter does not discover concrete ROS topic paths. It reads named
samples from `DataSession`, using the same names produced by the official
RealSense wrapper expansion: `d405_color`, `d405_color_info`, `d405_infra1`,
`d405_infra2`, `d405_infra1_info`, and `d405_infra2_info`, plus
`robot_state`, `left_eef`, and `right_eef`.

## GPU Side

Start Fast Foundation Stereo:

```bash
cd /path/to/PrometheusV3/third_party/fast-foundationstereo
python scripts/serve_depth.py \
  --model_dir /path/to/fast-foundationstereo/weights/model_best_bp2_serialize.pth \
  --host 127.0.0.1 \
  --port 2172 \
  --device cuda \
  --replace-existing
```

Start HumanEgo:

```bash
cd /path/to/PrometheusV3/third_party/HumanEgo
python -m inference.serve_policy \
  --cfg cfg/inference/prometheus_arx_d405.yaml \
  --host 127.0.0.1 \
  --port 2171 \
  --device cuda \
  --replace-existing
```

These services do not own robot hardware and do not send robot commands.

## Robot Side

Use the PrometheusV3 environment:

```bash
source /home/wujie/miniconda3/etc/profile.d/conda.sh
conda activate prometheusv3
cd /home/wujie/PrometheusV3
```

The workflow loads `robot=x5_2_humanego` by default:

```bash
python -m prometheus.workflows.rollout_sync --config-name rollout_humanego_sync
```

Use `x5_1` with the same policy names by overriding only the robot preset:

```bash
python -m prometheus.workflows.rollout_sync --config-name rollout_humanego_sync robot=x5_1_humanego
```

The default keeps `scheduler.dry_run=true`. Do not set it to `false` until the
policy payload and scheduled actions have been inspected.

Run hardware live checks independently before rollout:

```bash
PROMETHEUS_LIVE_ARX5=1 PROMETHEUS_ROBOT_CONFIG=x5_2 python tests/test_arx5_live.py
PROMETHEUS_LIVE_REALSENSE=1 PROMETHEUS_REALSENSE_CONFIG=x5_2_humanego python tests/test_realsense_live.py
```

The config carries separate robot/control and sensor ROS domain IDs. The
ROS-to-`DataSession` ingress bridge is documented in
`docs/ROS_DATA_BRIDGE.md`. Do not treat dry-run rollout as fully deployed
until the bridge has been verified on the target robot host.

## Safety

Keep `dry_run=true` until the printed action chunks are reasonable. HumanEgo
does not home the robot or validate the physical scene. The operator must place
the robot in a safe start pose and supervise any motion.
