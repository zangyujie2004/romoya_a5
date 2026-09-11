"""A5 stationary teach experiment: direct SDK versus ZMQ sensor reception.

This is a hardware experiment: connect() homes both arms before teach mode.
No policy actions are sent. Ctrl-C ends the experiment and parks using the
configured shutdown pose before closing the driver.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.a5_sensor_load import load_config, receiver, write_json

PHASES = {'baseline', 'shared-zmq-cameras', 'shared-zmq-all', 'split-zmq-all', 'direct-cameras', 'direct-all'}


def snapshot_processes(root_pid):
    processes = {}
    for path in Path('/proc').glob('[0-9]*'):
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            processes[int(path.name)] = (int(fields[1]), fields, path)
        except (OSError, ValueError, IndexError):
            pass
    selected = {root_pid}
    while True:
        new = {pid for pid, (parent, _, _) in processes.items() if parent in selected}
        if new.issubset(selected):
            break
        selected |= new
    result = {}
    for pid in selected:
        if pid not in processes:
            continue
        _, fields, path = processes[pid]
        try:
            masks = {}
            for task in (path / 'task').iterdir():
                try:
                    mask = ','.join(map(str, sorted(os.sched_getaffinity(int(task.name)))))
                    masks[mask] = masks.get(mask, 0) + 1
                except ProcessLookupError:
                    pass
            result[pid] = {'cpu_ticks': int(fields[11])+int(fields[12]), 'rss_pages': int(fields[21]),
                           'thread_affinities': masks, 'command': (path/'cmdline').read_bytes().replace(b'\0', b' ').decode()}
        except (OSError, ValueError):
            pass
    return result


class Monitor:
    def __init__(self, driver, output):
        self.driver = driver
        self.samples = (output / 'joints.jsonl').open('w', buffering=1)
        self.processes = (output / 'processes.jsonl').open('w', buffering=1)
        self.last_process = 0
        self.reset()

    def reset(self):
        self.rows = []

    def tick(self, phase, *, measured=False):
        import numpy as np
        begin = time.monotonic()
        state = self.driver.read_state()
        q = np.asarray(state.qpos)
        if q.shape != (14,) or not np.isfinite(q).all():
            raise RuntimeError('invalid robot feedback')
        row = {'time_ns': time.time_ns(), 'monotonic': begin, 'phase': phase,
               'qpos': q.tolist(), 'qvel': np.asarray(state.qvel).tolist(),
               'read_ms': (time.monotonic()-begin)*1000}
        self.samples.write(json.dumps(row)+'\n')
        if measured:
            self.rows.append(row)
        if begin - self.last_process >= 1:
            self.processes.write(json.dumps({'time_ns': time.time_ns(), 'phase': phase,
                                            'processes': snapshot_processes(os.getpid())})+'\n')
            self.last_process = begin
        time.sleep(max(0, .01-(time.monotonic()-begin)))

    def summary(self):
        import numpy as np
        q = np.array([r['qpos'] for r in self.rows])
        times = np.array([r['monotonic'] for r in self.rows])
        if len(q)<2:
            return {'samples': len(q)}
        delta = np.abs(np.diff(q, axis=0))
        return {'samples':len(q), 'duration_s':float(times[-1]-times[0]),
                'max_step_rad':delta.max(axis=0).tolist(), 'range_rad':np.ptp(q,axis=0).tolist(),
                'steps_over_0_05_rad':(delta>.05).sum(axis=0).tolist(),
                'poll_gap_p99_ms':float(np.quantile(np.diff(times),.99)*1000),
                'poll_gap_max_ms':float(np.diff(times).max()*1000)}

    def close(self):
        self.samples.close()
        self.processes.close()


def stop_sensors(proc):
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def run_phase(name, directory, driver, monitor, args, config_path):
    directory.mkdir()
    monitor.reset()
    proc = data = log = None
    summary = {'phase':name, 'status':'starting'}
    try:
        if name != 'baseline':
            transport = 'direct' if name.startswith('direct') else 'zmq'
            sensor_set = 'cameras' if name.endswith('cameras') else 'all'
            cfg = load_config(config_path, sensor_set)
            cmd = [sys.executable, '-u', str(ROOT/'scripts/a5_sensor_load.py'), '--config', str(config_path),
                   '--output', str(directory), '--transport', transport, '--sensor-set', sensor_set,
                   '--duration-s', str(args.phase_seconds+180)]
            if name.startswith('split'):
                cmd.append('--subscribe')
            log = (directory/'sensors.log').open('w')
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if name.startswith('shared'):
                data = receiver(cfg)
                data.start()
            timeout = time.monotonic()+70
            while not (directory/'READY.json').exists():
                if proc.poll() is not None:
                    raise RuntimeError(f'sensor process exited {proc.returncode}; see {directory}/sensors.log')
                if time.monotonic()>timeout:
                    raise TimeoutError('sensor readiness timeout')
                monitor.tick(name+':startup')
            if data is not None:
                while not data.ready() or not data.numpy.buffer.ready():
                    if time.monotonic()>timeout:
                        raise TimeoutError('shared ZMQ receiver readiness timeout')
                    monitor.tick(name+':subscriber-startup')
        print(f'[MEASURE] {name}: {args.phase_seconds}s, keep arms untouched',flush=True)
        end = time.monotonic()+args.phase_seconds
        last_report = time.monotonic()
        first_counts = dict(data.experiment_counts) if data is not None else {}
        measurement_start = time.monotonic()
        while time.monotonic()<end:
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f'sensor process failed with {proc.returncode}')
            if data is not None:
                latest = data.latest_numpy()
                stale = [k for k,s in latest.items() if time.time_ns()-s.recv_ns>2_000_000_000]
                if stale:
                    raise RuntimeError(f'ZMQ streams stopped: {stale}')
            monitor.tick(name, measured=True)
            if time.monotonic()-last_report>10:
                info=monitor.summary()
                print(f"[PROGRESS] {name}: wrist ranges={info['range_rad'][5]:.4f}, {info['range_rad'][12]:.4f} rad",flush=True)
                last_report=time.monotonic()
        summary.update(monitor.summary(),status='completed')
        if data is not None:
            summary['receiver_status']=data.status().as_dict()
            elapsed=time.monotonic()-measurement_start
            summary['received_hz']={k:(v-first_counts.get(k,0))/elapsed for k,v in data.experiment_counts.items()}
        print('[RESULT] '+json.dumps(summary),flush=True)
        return summary
    except BaseException as exc:
        summary.update(monitor.summary(),status='failed',error=repr(exc))
        raise
    finally:
        try:
            if data is not None:
                data.close()
        finally:
            stop_sensors(proc)
            if log is not None:
                log.close()
            write_json(directory/'summary.json',summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'prometheus/configs/robot/a5_drag.yaml'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--phases',default='baseline,shared-zmq-cameras,direct-cameras,direct-all,split-zmq-all,shared-zmq-all,baseline')
    parser.add_argument('--phase-seconds',type=float,default=60)
    parser.add_argument('--cpu-mode',choices=('all','isolated'),default='all')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    phases=args.phases.split(',')
    if set(phases)-PHASES or args.phase_seconds<=0:
        parser.error('invalid phases or duration')
    from omegaconf import OmegaConf
    from prometheus.nodes.robots.a5 import A5DragDriver, runtime_config
    from prometheus.utils.cpu_affinity import configure_process_cpu_affinity
    available=sorted(os.sched_getaffinity(0))
    cfg=load_config(args.config,'all')
    if args.cpu_mode=='all':
        configure_process_cpu_affinity(available)
        cfg['robot']['control_cpu_affinity']=None
    else:
        collect=OmegaConf.to_container(OmegaConf.load(ROOT/'prometheus/configs/collect_a5.yaml'),resolve=False)
        data_cpus=collect['workflow']['data_cpu_affinity']
        if not set(data_cpus).issubset(available):
            raise RuntimeError('configured data CPUs are unavailable')
        configure_process_cpu_affinity(data_cpus)
    # RealSense SDK serial numbers differ from USB descriptor serial numbers.
    # Enumerate in a subprocess so SDK discovery threads do not enter the arm process.
    discovery=subprocess.check_output([sys.executable,'-c',
        'import json,pyrealsense2 as rs; print(json.dumps([d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]))'],text=True,timeout=15)
    serials=json.loads(discovery)
    missing=set(cfg['cameras']['serial_no'].values())-set(serials)
    if missing:
        raise RuntimeError(f'configured RealSense SDK serials not found: {missing}; detected {serials}')
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    config_path=output/'robot.yaml'
    OmegaConf.save(OmegaConf.create(cfg),config_path)
    write_json(output/'experiment.json',{'cpus':sorted(os.sched_getaffinity(0)), 'cpu_mode':args.cpu_mode,
        'control_cpus':cfg['robot']['control_cpu_affinity'], 'phases':phases,'phase_seconds':args.phase_seconds,
        'onnx_inference_threads':cfg['tactile']['inference_threads'],
        'note':'Direct mode omits ZMQ serialization/transport/decoding; differences alone do not prove DDS is at fault.'})
    kwargs=runtime_config(cfg)['driver'];kwargs.pop('_target_')
    driver=A5DragDriver(**kwargs)
    if args.check_only:
        driver._import_single_arm()
        print(json.dumps({'config':str(config_path),'driver':driver.status()},default=str),flush=True)
        return
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    monitor=None
    ready=False
    try:
        print(f'[CONNECT] CPU mode {args.cpu_mode}; homing then teach in 3 seconds',flush=True)
        time.sleep(3)
        driver.connect()
        ready=True
        monitor=Monitor(driver,output)
        print('[READY] dual arms in teach; no policy commands will be sent',flush=True)
        for i,name in enumerate(phases):
            run_phase(name,output/f'{i:02d}_{name}',driver,monitor,args,config_path)
    finally:
        try:
            if ready:
                print('[EXIT] moving to configured shutdown pose',flush=True)
                driver.move_to_shutdown_safe_position()
        finally:
            driver.close()
            if monitor is not None:
                monitor.close()


if __name__=='__main__':
    main()
