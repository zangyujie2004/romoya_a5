"""Sensor-only load for A5 jitter experiments. Never opens a robot CAN device."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def load_config(path, sensor_set):
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if sensor_set == 'cameras':
        cfg['tactile']['serial_no'] = {}
    return cfg


def sensor_contract(cfg):
    from prometheus.nodes.sensors import realsense, xense
    return ({**realsense.runtime_config(cfg), **xense.runtime_config(cfg)},
            {**realsense.streams(cfg), **xense.streams(cfg)})


def receiver(cfg):
    from prometheus.sessions.data import DataSession
    _, streams = sensor_contract(cfg)
    class CountingDataSession(DataSession):
        def __init__(self, *args, **kwargs):
            self.experiment_counts = {}
            super().__init__(*args, **kwargs)

        def ingest(self, name, *args, **kwargs):
            result = super().ingest(name, *args, **kwargs)
            self.experiment_counts[name] = self.experiment_counts.get(name, 0) + 1
            return result
    return CountingDataSession(streams, history=64,
                       bridge={'enabled': True, 'name': 'jitter_sensor_receiver', 'qos_depth': 10,
                               'wait_all_streams': True, 'sensor_qos': 'sensor'},
                       numpy={'enabled': True, 'history': 64}, recording={'enabled': False})


def direct_worker(kind, name, serial, cfg, stop, events):
    """Match SDK capture, resize and flow assembly; omit ZMQ serialization/IPC."""
    import numpy as np
    count = 0
    last_report = time.monotonic()
    pipeline = sensor = None
    pipeline_started = False
    previous_token = None
    unique = 0
    try:
        if kind == 'camera':
            import pyrealsense2 as rs
            from prometheus.nodes.sensors.realsense import _resize_image, normalize_resize
            settings = cfg['cameras']
            resize = normalize_resize(settings['prepub_resize'])
            pipeline = rs.pipeline()
            rs_cfg = rs.config()
            rs_cfg.enable_device(serial)
            rs_cfg.enable_stream(rs.stream.color, settings['capture_width'], settings['capture_height'],
                                 rs.format.bgr8, settings['capture_hz'])
            if settings['ir']:
                for index in (1, 2):
                    rs_cfg.enable_stream(rs.stream.infrared, index, settings['capture_width'],
                                         settings['capture_height'], rs.format.y8, settings['capture_hz'])
            pipeline.start(rs_cfg)
            pipeline_started = True
        else:
            from prometheus.nodes.sensors.xense import (
                preload_xense_gpu_libraries, create_xense_sensor, read_xense_frame,
                _publish_tactile_image, normalize_resize,
            )
            settings = cfg['tactile']
            preload_xense_gpu_libraries(required=settings['require_gpu'])
            from xensesdk import Sensor
            sensor, _ = create_xense_sensor(Sensor, serial, inference_threads=settings['inference_threads'])
            if settings['require_gpu'] and not sensor.infer_engine_using_gpu:
                raise RuntimeError('Xense fell back to CPU')
            resize = normalize_resize(settings['prepub_resize'])
            # Reuse production image preparation, with a local copy in place of CvBridge.
            from types import SimpleNamespace
            class Bridge:
                def cv2_to_imgmsg(self, array, encoding):
                    return SimpleNamespace(header=SimpleNamespace(), data=array.copy())
            class Sink:
                def publish(self, msg):
                    pass
            bridge, sink = Bridge(), Sink()
        deadline = time.monotonic()
        while not stop.is_set():
            if kind == 'camera':
                frames = pipeline.wait_for_frames(timeout_ms=2000)
                frame = frames.get_color_frame()
                if not frame:
                    raise RuntimeError('missing RGB frame')
                token = frame.get_frame_number()
                image = _resize_image(np.asanyarray(frame.get_data()), resize).copy()
                if settings['ir']:
                    for index in (1, 2):
                        _resize_image(np.asanyarray(frames.get_infrared_frame(index).get_data()), resize).copy()
            else:
                outputs = read_xense_frame(sensor, Sensor.OutputType)
                token = str(outputs['sdk_timestamp'])
                if settings['publish_flow']:
                    points = np.concatenate((np.asarray(outputs['mesh']).reshape(-1, 3),
                                             np.asarray(outputs['flow']).reshape(-1, 3)), axis=1).astype(np.float32)
                    if points.shape != (700, 6) or not np.isfinite(points).all():
                        raise RuntimeError('invalid tactile flow')
                for key, enabled in (('difference', settings['publish_difference']), ('raw', settings['publish_raw'])):
                    if enabled:
                        _publish_tactile_image(bridge, sink, name, key, outputs[key], None, resize)
            count += 1
            unique += int(token != previous_token)
            previous_token = token
            now = time.monotonic()
            if count == 1 or now - last_report >= 1:
                events.put({'name': f'{kind}:{name}', 'count': count, 'unique': unique, 'time_ns': time.time_ns(),
                            'cpu_affinity': sorted(os.sched_getaffinity(0))})
                last_report = now
            deadline += 1.0 / settings['publish_hz']
            stop.wait(max(0, deadline - time.monotonic()))
    except BaseException as exc:
        events.put({'name': f'{kind}:{name}', 'error': repr(exc)})
        raise
    finally:
        if sensor is not None:
            sensor.release()
        if pipeline_started:
            pipeline.stop()


def run(args):
    cfg = load_config(args.config, args.sensor_set)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stop_requested = False
    def request_stop(*_):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    hardware = data = None
    processes = []
    ctx = mp.get_context('spawn')
    stop = ctx.Event()
    events = ctx.Queue()
    status = {}
    metrics = (output / 'sensor_metrics.jsonl').open('w', buffering=1)
    deadline = time.monotonic() + args.duration_s
    try:
        if args.transport == 'direct':
            for kind, mapping in (('camera', cfg['cameras']['serial_no']), ('tactile', cfg['tactile']['serial_no'])):
                for name, serial in mapping.items():
                    proc = ctx.Process(target=direct_worker, args=(kind, name, serial, cfg, stop, events), name=name)
                    proc.start()
                    processes.append(proc)
            expected = len(processes)
        else:
            from prometheus.sessions.hardware import HardwareSession
            sensors, streams = sensor_contract(cfg)
            hardware = HardwareSession(run_id=output.name, runtime_dir=output, robot={}, sensors=sensors, streams=streams)
            hardware.start()
            if args.subscribe:
                data = receiver(cfg)
                data.start()
            hardware.wait_ready(45)
            if data is not None:
                data.wait_ready(20)
            write_json(output / 'READY.json', {'pid': os.getpid(), 'transport': args.transport})
        began = time.monotonic()
        while not stop_requested and time.monotonic() < deadline:
            if args.transport == 'direct':
                while True:
                    try:
                        item = events.get_nowait()
                    except queue.Empty:
                        break
                    if 'error' in item:
                        raise RuntimeError(str(item))
                    status[item['name']] = item
                if any(p.exitcode is not None for p in processes):
                    raise RuntimeError('sensor worker exited unexpectedly')
                if len(status) == expected and not (output / 'READY.json').exists():
                    write_json(output / 'READY.json', {'pid': os.getpid(), 'transport': args.transport})
                if time.monotonic() - began > 45 and len(status) != expected:
                    raise TimeoutError('direct sensors not ready')
                if any(time.time_ns()-s['time_ns'] > 5_000_000_000 for s in status.values()):
                    raise TimeoutError('direct sensor stalled')
            else:
                status = {name: node.status() for name, node in hardware.sensor_nodes.items()}
                for name, node in hardware.sensor_nodes.items():
                    if any(not p.is_alive() for p in node._procs.values()):
                        raise RuntimeError(f'{name} ZMQ publisher exited')
                if data is not None:
                    status['receiver'] = data.status().as_dict()
                    status['received_counts'] = dict(data.experiment_counts)
                    stale = [k for k,s in data.latest_numpy().items() if time.time_ns()-s.recv_ns>2_000_000_000]
                    if stale:
                        raise TimeoutError(f'ZMQ receiver stalled: {stale}')
            write_json(output / 'status.json', status)
            metrics.write(json.dumps({'time_ns': time.time_ns(), 'status':status}, default=str)+'\n')
            time.sleep(.2)
        if not stop_requested and not (output / 'READY.json').exists():
            raise TimeoutError('duration elapsed before all sensors became ready')
    finally:
        metrics.close()
        stop.set()
        for proc in processes:
            proc.join(4)
            if proc.is_alive():
                proc.terminate()
                proc.join(2)
            if proc.is_alive():
                proc.kill()
                proc.join()
        try:
            if data is not None:
                data.close()
        finally:
            if hardware is not None:
                hardware.stop()
                hardware.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--transport', choices=('direct', 'zmq'), required=True)
    parser.add_argument('--sensor-set', choices=('cameras', 'all'), default='all')
    parser.add_argument('--subscribe', action='store_true', help='ZMQ receiver in sensor process, separate from robot')
    parser.add_argument('--duration-s', type=float, default=600)
    run(parser.parse_args())
