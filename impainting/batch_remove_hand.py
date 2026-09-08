#!/usr/bin/env python3
"""Batch scheduler for final Remove_Hand pipeline."""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SCRIPT = ROOT / "remove_hand.py"
DEFAULT_MODELS = ROOT / "model_ckpt"
DEFAULT_PROPAINTER = ROOT / "third_party" / "ProPainter"
DEFAULT_PYTHON = Path(sys.executable)


@dataclass(frozen=True)
class Job:
    task_root: Path
    input_video: Path
    output_video: Path
    metadata_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a list of task roots and write base_0_rgb_remove_hand.mp4 next to each source video."
    )
    parser.add_argument("--task-list", action="append", default=[], help="Task root to process. Repeat this option for multiple tasks.")
    parser.add_argument("--task-list-file", default="", help="Optional text file with task roots, one per line.")
    parser.add_argument("--script", default=str(DEFAULT_SCRIPT))
    parser.add_argument("--models-root", default=str(DEFAULT_MODELS))
    parser.add_argument("--propainter-root", default=str(DEFAULT_PROPAINTER))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--gpus", default="auto", help="auto, or comma list such as 0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--input-name", default="base_0_rgb.mp4")
    parser.add_argument("--output-name", default="base_0_rgb_remove_hand.mp4")
    parser.add_argument("--json-dir-name", default="remove_hand")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-mask-video", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--box-threshold", type=float, default=0.22)
    parser.add_argument("--text-threshold", type=float, default=0.22)
    parser.add_argument("--roi-bottom-ratio", type=float, default=0.38)
    parser.add_argument("--dino-interval", type=int, default=1)
    parser.add_argument("--box-expand-ratio", type=float, default=0.20)
    parser.add_argument("--min-box-margin", type=int, default=12)
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--mask-dilation", type=int, default=5)
    parser.add_argument("--temporal-flow-width", type=int, default=320)
    parser.add_argument("--mask-mode", choices=("strict_color", "color_shadow"), default="strict_color")
    parser.add_argument("--neighbor-length", type=int, default=8)
    parser.add_argument("--ref-stride", type=int, default=5)
    parser.add_argument("--subvideo-length", type=int, default=20)
    parser.add_argument("--raft-iter", type=int, default=10)
    parser.add_argument("--propainter-mask-dilation", type=int, default=1)
    return parser.parse_args()


def detect_gpus(spec: str) -> list[str]:
    if spec != "auto":
        gpus = [x.strip() for x in spec.split(",") if x.strip()]
        if not gpus:
            raise SystemExit("--gpus is empty")
        return gpus
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        gpus = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        gpus = []
    if not gpus:
        raise SystemExit("No GPU detected. Use --gpus 0 or check nvidia-smi.")
    return gpus


def read_task_roots(paths: list[str], list_file: str) -> list[Path]:
    roots: list[Path] = []
    for item in paths:
        line = item.strip()
        if line:
            roots.append(Path(line).expanduser())
    if list_file:
        path = Path(list_file)
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            roots.append(Path(line).expanduser())
    if not roots:
        raise SystemExit("No task roots. Use --task-list /path/to/task, repeated as needed.")
    return roots


def discover_jobs(task_roots: list[Path], input_name: str, output_name: str, json_dir_name: str, overwrite: bool) -> list[Job]:
    jobs: list[Job] = []
    for root in task_roots:
        if not root.is_dir():
            print(f"[skip] missing task root: {root}", file=sys.stderr)
            continue
        for video in sorted(root.glob(f"*/camera/{input_name}")):
            output_video = video.with_name(output_name)
            metadata_dir = video.parent / json_dir_name
            if output_video.resolve() == video.resolve():
                raise SystemExit(f"Refuse to overwrite source video: {video}")
            if output_video.exists() and not overwrite:
                print(f"[skip] exists: {output_video}")
                continue
            jobs.append(Job(root, video, output_video, metadata_dir))
    return jobs


def build_command(args: argparse.Namespace, job: Job) -> list[str]:
    cmd = [
        args.python,
        args.script,
        "--input", str(job.input_video),
        "--models-root", args.models_root,
        "--propainter-root", args.propainter_root,
        "--output-dir", str(job.metadata_dir),
        "--output-name", args.output_name,
        "--box-threshold", str(args.box_threshold),
        "--text-threshold", str(args.text_threshold),
        "--roi-bottom-ratio", str(args.roi_bottom_ratio),
        "--dino-interval", str(args.dino_interval),
        "--box-expand-ratio", str(args.box_expand_ratio),
        "--min-box-margin", str(args.min_box_margin),
        "--min-area", str(args.min_area),
        "--mask-dilation", str(args.mask_dilation),
        "--temporal-flow-width", str(args.temporal_flow_width),
        "--mask-mode", args.mask_mode,
        "--neighbor-length", str(args.neighbor_length),
        "--ref-stride", str(args.ref_stride),
        "--subvideo-length", str(args.subvideo_length),
        "--raft-iter", str(args.raft_iter),
        "--propainter-mask-dilation", str(args.propainter_mask_dilation),
        "--progress-every", str(args.progress_every),
    ]
    if args.keep_mask_video:
        cmd.append("--keep-mask-video")
    return cmd


def run_one(args: argparse.Namespace, job: Job, gpu: str, worker_name: str, results: list[dict[str, object]], lock: threading.Lock) -> None:
    start = time.time()
    episode = job.input_video.parents[1].name
    task = job.task_root.name
    job.metadata_dir.mkdir(parents=True, exist_ok=True)
    log_path = job.metadata_dir / f"{task}_{episode}_gpu{gpu}_{worker_name}.log"
    temp_output = job.metadata_dir / args.output_name
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    # The interpreter selected by --python must use its own environment. In
    # particular, never mix binary packages from the project legacy Python
    # 3.10 virtualenv into a Python 3.11 process via PYTHONPATH.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.pop("VIRTUAL_ENV", None)
    cmd = build_command(args, job)
    print(f"[{worker_name}] gpu={gpu} start {job.input_video}")
    rc = 0
    try:
        if temp_output.exists():
            temp_output.unlink()
        with log_path.open("w") as log:
            log.write("command: " + " ".join(cmd) + "\n")
            log.write(f"input: {job.input_video}\n")
            log.write(f"temp_output: {temp_output}\n")
            log.write(f"final_output: {job.output_video}\n")
            log.write(f"gpu: {gpu}\n\n")
            log.flush()
            completed = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            rc = completed.returncode
        if rc == 0:
            if not temp_output.is_file():
                rc = 998
            else:
                tmp_commit = job.output_video.with_suffix(job.output_video.suffix + ".uploading")
                if tmp_commit.exists():
                    tmp_commit.unlink()
                shutil.copy2(temp_output, tmp_commit)
                tmp_commit.replace(job.output_video)
                temp_output.unlink()
    except Exception as exc:
        rc = 999
        with log_path.open("a") as log:
            log.write(f"\nSCHEDULER_ERROR: {type(exc).__name__}: {exc}\n")
    elapsed = time.time() - start
    ok = rc == 0 and job.output_video.is_file()
    with lock:
        results.append({
            "ok": ok,
            "returncode": rc,
            "gpu": gpu,
            "input": str(job.input_video),
            "output": str(job.output_video),
            "metadata_dir": str(job.metadata_dir),
            "log": str(log_path),
            "elapsed_s": elapsed,
        })
    print(f"[{worker_name}] gpu={gpu} done ok={ok} rc={rc} elapsed={elapsed:.1f}s {job.output_video}")


def worker_loop(args: argparse.Namespace, gpu: str, worker_idx: int, jobs: "queue.Queue[Job]", results: list[dict[str, object]], lock: threading.Lock) -> None:
    worker_name = f"gpu{gpu}_w{worker_idx}"
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        try:
            run_one(args, job, gpu, worker_name, results, lock)
        finally:
            jobs.task_done()


def main() -> None:
    args = parse_args()
    if args.workers_per_gpu < 1:
        raise SystemExit("--workers-per-gpu must be >= 1")
    for required in (Path(args.script), Path(args.models_root), Path(args.propainter_root), Path(args.python)):
        if not required.exists():
            raise SystemExit(f"Missing required path: {required}")
    gpus = detect_gpus(args.gpus)
    jobs_list = discover_jobs(
        read_task_roots(args.task_list, args.task_list_file),
        args.input_name,
        args.output_name,
        args.json_dir_name,
        args.overwrite,
    )
    print(f"videos_to_process={len(jobs_list)} gpus={','.join(gpus)} workers_per_gpu={args.workers_per_gpu}")
    for job in jobs_list:
        print(f"job input={job.input_video} output={job.output_video} json={job.metadata_dir}")
    if args.dry_run or not jobs_list:
        return
    jobs_q: "queue.Queue[Job]" = queue.Queue()
    for job in jobs_list:
        jobs_q.put(job)
    results: list[dict[str, object]] = []
    lock = threading.Lock()
    threads: list[threading.Thread] = []
    for gpu in gpus:
        for worker_idx in range(args.workers_per_gpu):
            thread = threading.Thread(target=worker_loop, args=(args, gpu, worker_idx, jobs_q, results, lock), daemon=False)
            thread.start()
            threads.append(thread)
    for thread in threads:
        thread.join()
    summary = {
        "gpus": gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "jobs": sorted(results, key=lambda item: str(item["input"])),
        "ok": sum(1 for item in results if item["ok"]),
        "failed": sum(1 for item in results if not item["ok"]),
    }
    summary_path = Path("/mnt/data/ts/remove_hand") / f"batch_remove_hand_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"summary={summary_path}")
    print(f"ok={summary['ok']} failed={summary['failed']}")
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
