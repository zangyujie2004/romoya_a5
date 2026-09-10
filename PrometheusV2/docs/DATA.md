# Data Recording

Prometheus records raw data streams, not selected aligned data views.

Runtime path:

```text
HardwareSession.streams  -> DataSession ROS subscribers
DataSession.ingest()     -> online buffer + async raw recording queue
Policy ActionChunk       -> DataSession.record_action_chunk()
Workflow/Event input     -> DataSession.record_event()
```

`DataSession.latest()`, `frame()`, and `window()` are read views. Recording
must not hook into those methods, because they only represent data points a
workflow happened to consume.

Each episode uses a raw layout close to the hardware sources:

```text
episodes/0001/
  camera/
    <camera>_color.mp4
    <camera>_infra1.mkv
    <camera>_infra2.mkv
    timestamp.pkl
    metadata.json
  tactile/
  robot/
    robot_state_dict.pkl
  action/
    action_chunk_dict.pkl
  event/
    event_dict.pkl
  manifests/
    camera.json
    robot.json
    action.json
    event.json
```

Camera streams are committed as complete physical-camera bundles. Color,
infra1, and infra2 frames with the same source timestamp share one timestamp
entry under `camera/timestamp.pkl::<camera>`.

Robot state is committed by source timestamp across `robot_state`, `left_eef`,
and `right_eef`. Missing fields are stored as `None` at finalize time instead
of fake values.

Action chunks and workflow events are recorded as their own raw modalities.
Tactile storage is intentionally left empty until tactile hardware is wired
into this framework.

`DataSession` remains the only online timestamp-alignment implementation.
Offline export should reuse the same alignment logic over raw episode payloads
instead of reimplementing nearest-sample or slop behavior.
