# Policy Integration

Policies are workflow tools. A policy reads workflow data and returns one
`ActionChunk`; it does not own robot hardware or sensor nodes.

The normal runtime path is:

```text
workflow
  -> DataSession
  -> PolicySession.policy.infer()
  -> ActionChunk
  -> ActionScheduler
  -> RobotControlClient
```

`PolicySession` is only a lifecycle wrapper. Use it when a policy owns a
resource such as a websocket connection, model process, GPU server, or remote
runtime. Do not put rollout procedure logic inside a policy session.

Policy implementations should keep three boundaries explicit:

- data stream names and alignment semantics are public config;
- model/server-specific payloads stay inside the concrete policy adapter;
- action validation, filtering, timing, and robot send calls stay in
  `ActionScheduler`.

Remote policies should use `prometheus.utils.remote.RemoteInferenceClient`.
The remote client is transport only: it sends one request dict and receives one
response dict. Request schemas belong to the concrete policy adapter.

Hydra config is the policy public interface:

```yaml
policy:
  _target_: prometheus.policy.some_policy.SomePolicy
  inputs:
    anchor: d405_color
    slop_ms: 100.0
  remote:
    server: ws://127.0.0.1:2171
```

Do not let a policy discover ROS topics, robot ids, or config files on its own.
Those belong to the workflow, `HardwareSession`, and `DataSession`.
