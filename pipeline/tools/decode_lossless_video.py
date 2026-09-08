#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    codec: str
    pixel_format: str
    width: int
    height: int
    fps: float


def environment_executable(name: str) -> str:
    candidate = Path(sys.executable).with_name(name)
    if candidate.is_file():
        return str(candidate)
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"{name} executable was not found")
    return executable


def parse_frame_rate(value: str) -> float:
    numerator, separator, denominator = value.partition("/")
    if not separator:
        return float(numerator)
    denominator_value = float(denominator)
    return 0.0 if denominator_value == 0 else float(numerator) / denominator_value


def probe_video(path: str | Path) -> VideoInfo:
    video_path = Path(path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    result = subprocess.run(
        [
            environment_executable("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,avg_frame_rate",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream in {video_path}")
    stream = streams[0]
    return VideoInfo(
        path=video_path,
        codec=str(stream["codec_name"]),
        pixel_format=str(stream.get("pix_fmt", "")),
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=parse_frame_rate(str(stream.get("avg_frame_rate", "0/1"))),
    )


def infer_stream_kind(path: str | Path) -> str:
    name = Path(path).stem.lower()
    if name.endswith("_rgb"):
        return "rgb"
    if name.endswith(("_ir_left", "_ir_right")):
        return "ir"
    if name.endswith(("_difference", "_rectify")):
        return "tactile"
    raise ValueError(
        f"Cannot infer stream kind from {Path(path).name}; "
        "pass --kind rgb, ir, or tactile"
    )


def read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def iter_video_frames(
    path: str | Path,
    *,
    kind: str = "auto",
    color_order: str = "rgb",
    require_lossless: bool | None = None,
) -> Iterator[np.ndarray]:
    """Decode one Prometheus camera or tactile video without resizing.

    IR frames are yielded as uint8 arrays with shape [height, width].
    RGB and tactile frames are yielded as uint8 [height, width, 3] in RGB by
    default. Selecting BGR keeps FFmpeg ``bgr24`` channel order.

    FFV1 IR/tactile input is decoded pixel-exactly. Camera RGB uses lossy HEVC,
    so decoding recovers the stored frame but not the pre-encoding pixels.
    """

    info = probe_video(path)
    if kind == "auto":
        kind = infer_stream_kind(info.path)
    if kind not in {"rgb", "ir", "tactile"}:
        raise ValueError("kind must be auto, rgb, ir, or tactile")
    if color_order not in {"bgr", "rgb"}:
        raise ValueError("color_order must be bgr or rgb")

    if require_lossless is None:
        require_lossless = kind in {"ir", "tactile"}
    if require_lossless and info.codec != "ffv1":
        raise ValueError(
            f"{info.path} uses codec {info.codec!r}; exact FFV1 decoding required"
        )
    if kind == "rgb" and info.codec == "ffv1":
        raise ValueError(
            f"{info.path} is marked as RGB but uses FFV1; "
            "check --kind or the filename"
        )

    is_ir = kind == "ir"
    output_pixel_format = "gray" if is_ir else "bgr24"
    channels = 1 if is_ir else 3
    frame_bytes = info.width * info.height * channels
    command = [
        environment_executable("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(info.path),
        "-map",
        "0:v:0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        output_pixel_format,
        "pipe:1",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=frame_bytes,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError(f"Failed to open FFmpeg pipes for {info.path}")

    completed = False
    try:
        while True:
            payload = read_exact(process.stdout, frame_bytes)
            if not payload:
                break
            if len(payload) != frame_bytes:
                raise RuntimeError(
                    f"Truncated frame in {info.path}: "
                    f"expected {frame_bytes} bytes, got {len(payload)}"
                )
            shape = (
                (info.height, info.width)
                if is_ir
                else (info.height, info.width, 3)
            )
            frame = np.frombuffer(payload, dtype=np.uint8).reshape(shape).copy()
            if not is_ir and color_order == "rgb":
                frame = frame[..., ::-1].copy()
            yield frame

        stderr = process.stderr.read().decode(errors="replace").strip()
        returncode = process.wait()
        completed = True
        if returncode != 0:
            raise RuntimeError(
                f"FFmpeg decode failed for {info.path} "
                f"with returncode {returncode}: {stderr}"
            )
    finally:
        process.stdout.close()
        process.stderr.close()
        if not completed and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def cuda_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except OSError:
        return False


def _rawvideo_output_spec(kind: str, info: VideoInfo) -> tuple[str, int, tuple[int, ...]]:
    is_ir = kind == "ir"
    channels = 1 if is_ir else 3
    frame_bytes = info.width * info.height * channels
    pix_fmt = "gray" if is_ir else "bgr24"
    shape = (
        (info.height, info.width)
        if is_ir
        else (info.height, info.width, 3)
    )
    return pix_fmt, frame_bytes, shape


def _decode_rawvideo_bulk(
    path: str | Path,
    *,
    kind: str,
    backend: str,
    gpu_id: int,
) -> bytes:
    info = probe_video(path)
    if kind == "auto":
        kind = infer_stream_kind(info.path)
    pix_fmt, _frame_bytes, _shape = _rawvideo_output_spec(kind, info)

    command = [environment_executable("ffmpeg"), "-hide_banner", "-loglevel", "error"]
    if backend == "gpu":
        if kind != "rgb" or info.codec not in {"hevc", "h264"}:
            raise ValueError(
                f"GPU decode only supports HEVC/H264 RGB; got kind={kind!r} codec={info.codec!r}"
            )
        command.extend(
            [
                "-hwaccel",
                "cuda",
                "-hwaccel_device",
                str(gpu_id),
            ]
        )
    else:
        command.extend(["-threads", "0"])

    command.extend(
        [
            "-i",
            str(info.path),
            "-map",
            "0:v:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "pipe:1",
        ]
    )
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(
            f"FFmpeg decode failed ({backend}) for {info.path}: {stderr}"
        )
    return result.stdout


def read_video_array(
    path: str | Path,
    *,
    kind: str = "auto",
    color_order: str = "rgb",
    backend: str = "auto",
    gpu_id: int = 1,
) -> np.ndarray:
    """Decode a whole video into one ndarray.

    backend:
      - ``sw``: CPU ffmpeg (default for ``auto``; fastest on 640x480 in our tests)
      - ``gpu``: CUDA/NVDEC for HEVC/H264 RGB only
      - ``auto``: same as ``sw``
    """

    if backend not in {"auto", "sw", "gpu"}:
        raise ValueError("backend must be auto, sw, or gpu")
    if color_order not in {"bgr", "rgb"}:
        raise ValueError("color_order must be bgr or rgb")

    info = probe_video(path)
    resolved_kind = infer_stream_kind(info.path) if kind == "auto" else kind
    pix_fmt, frame_bytes, shape = _rawvideo_output_spec(resolved_kind, info)

    chosen = "gpu" if backend == "gpu" else "sw"
    if chosen == "gpu" and (resolved_kind != "rgb" or info.codec not in {"hevc", "h264"}):
        raise ValueError(
            f"GPU backend requested but {info.path.name} is {resolved_kind}/{info.codec}"
        )

    raw = _decode_rawvideo_bulk(
        path,
        kind=resolved_kind,
        backend=chosen,
        gpu_id=gpu_id,
    )
    frame_count = len(raw) // frame_bytes
    if frame_count == 0:
        raise RuntimeError(f"No frames decoded from {path}")
    if len(raw) % frame_bytes:
        raise RuntimeError(f"Truncated rawvideo payload from {path}")

    video = np.frombuffer(raw[: frame_count * frame_bytes], dtype=np.uint8).reshape(
        frame_count,
        *shape,
    )
    # Bulk decode is (T,H,W,3); single-frame helpers use ndim==3. Flip BGR→RGB on
    # the channel axis whenever color_order requests rgb (IR stays (T,H,W)).
    if color_order == "rgb" and video.ndim >= 3 and video.shape[-1] == 3:
        video = video[..., ::-1]
    return video.copy()


def benchmark_video_backends(
    path: str | Path,
    *,
    kind: str = "auto",
    gpu_id: int = 1,
    repeats: int = 1,
) -> dict[str, float]:
    """Return decode seconds for sw vs gpu (gpu omitted if unsupported)."""

    timings: dict[str, float] = {}
    for backend in ("sw", "gpu"):
        try:
            start = time.perf_counter()
            for _ in range(repeats):
                read_video_array(path, kind=kind, backend=backend, gpu_id=gpu_id)
            timings[backend] = (time.perf_counter() - start) / repeats
        except Exception:
            continue
    return timings


def iter_lossless_frames(
    path: str | Path,
    *,
    kind: str = "auto",
    color_order: str = "rgb",
) -> Iterator[np.ndarray]:
    """Decode an FFV1 IR or tactile stream pixel-exactly."""

    yield from iter_video_frames(
        path,
        kind=kind,
        color_order=color_order,
        require_lossless=True,
    )


def save_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    kind: str = "auto",
    output_format: str = "png",
    color_order: str = "rgb",
    start: int = 0,
    stop: int = -1,
    stride: int = 1,
) -> int:
    if start < 0:
        raise ValueError("start must be non-negative")
    if stop >= 0 and stop < start:
        raise ValueError("stop must be -1 or greater than or equal to start")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if output_format not in {"png", "npy"}:
        raise ValueError("output_format must be png or npy")

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    saved = 0
    for frame_index, frame in enumerate(
        iter_video_frames(
            video_path,
            kind=kind,
            color_order=color_order,
        )
    ):
        if frame_index < start:
            continue
        if stop >= 0 and frame_index >= stop:
            break
        if (frame_index - start) % stride:
            continue

        path = destination / f"{frame_index:05d}.{output_format}"
        if output_format == "npy":
            np.save(path, frame, allow_pickle=False)
        else:
            import cv2

            image = frame
            if frame.ndim == 3 and color_order == "rgb":
                image = frame[..., ::-1]
            if not cv2.imwrite(str(path), image):
                raise RuntimeError(f"Failed to write {path}")
        saved += 1
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode Prometheus RGB, IR, or tactile videos into PNG frames "
            "or NumPy arrays. IR/tactile FFV1 decoding is lossless; RGB HEVC "
            "decoding recovers the stored lossy frames."
        )
    )
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Default: <video parent>/<video stem>_decoded",
    )
    parser.add_argument(
        "--kind",
        choices=("auto", "rgb", "ir", "tactile"),
        default="auto",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("png", "npy"),
        default="png",
    )
    parser.add_argument(
        "--color_order",
        choices=("bgr", "rgb"),
        default="rgb",
        help="Channel order for decoded arrays (default rgb). PNG via OpenCV converts RGB→BGR at write.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=-1, help="Exclusive; -1 means end.")
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = probe_video(args.video)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else info.path.parent / f"{info.path.stem}_decoded"
    )
    print(
        f"[INFO] input={info.path} codec={info.codec} "
        f"pix_fmt={info.pixel_format} size={info.width}x{info.height} "
        f"fps={info.fps:.3f}"
    )
    kind = args.kind if args.kind != "auto" else infer_stream_kind(info.path)
    if kind == "rgb":
        print(
            "[WARN] RGB camera video is HEVC lossy; decoded frames match the "
            "stored video, not the original pre-encoding BGR pixels."
        )
    count = save_frames(
        info.path,
        output_dir,
        kind=kind,
        output_format=args.output_format,
        color_order=args.color_order,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
    )
    print(f"[OK] decoded {count} frames to {output_dir}")


if __name__ == "__main__":
    main()

# 