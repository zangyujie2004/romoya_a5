# ROS Data Bridge

Prometheus keeps robot/control traffic and sensor traffic in separate ROS
domains when running real hardware:

```text
robot/control domain -> RosDataBridge -> DataSession
sensor domain        -> RosDataBridge -> DataSession
```

The bridge is ingress only. It subscribes to ROS topics and calls
`DataSession.ingest()`. It does not republish sensor topics into the robot
domain.

## Why Split Domains

Robot state and control messages are small but timing-sensitive. Camera frames
are lower frequency, but each frame is large. If both live in one DDS domain,
the robot/control graph shares discovery, matching, queues, transport threads,
socket buffers, serialization pressure, and memory bandwidth with image traffic.

Separate domains keep the 120 Hz robot/control channel away from high-bandwidth
30 Hz camera streams. QoS and executors can reduce interference inside one
domain, but they are not hard isolation.

## Why The Bridge Preserves The Benefit

`DataSession` is not a ROS node. It is an in-process buffer with
`latest()`, `frame()`, and `window()`. The bridge receives samples from each
domain and stores the original ROS messages in that buffer.

This does not merge the ROS domains. Sensor frames still do not enter the
robot/control DDS graph. The unavoidable cost is only paid by the data and
policy process that actually needs the images.

## Runtime Rules

- One `rclpy.Context` is created per `ros_domain_id`.
- Each domain has its own node, executor, and spin thread.
- ROS callbacks only call `DataSession.ingest()`.
- Sensor topics use ROS sensor-data QoS; non-sensor topics use a small normal
  queue depth.
- Recording, image conversion, policy preprocessing, and alignment stay out of
  ROS callbacks.

The bridge is owned by `DataSession`. Workflow code and policies read only from
`DataSession`.
