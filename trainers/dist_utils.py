from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist


def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist_available_and_initialized():
        return 0
    return int(dist.get_rank())


def get_local_rank() -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return get_rank()


def get_world_size() -> int:
    if not is_dist_available_and_initialized():
        return 1
    return int(dist.get_world_size())


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist_available_and_initialized():
        dist.barrier()


def init_distributed(backend: str = "nccl") -> tuple[int, int, int, torch.device]:
    """Init process group from torchrun env. Returns rank, local_rank, world_size, device."""
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if world_size > 1 and not dist.is_initialized():
        if device.type == "cuda":
            try:
                dist.init_process_group(
                    backend=backend, init_method="env://", device_id=device
                )
            except TypeError:
                dist.init_process_group(backend=backend, init_method="env://")
        else:
            dist.init_process_group(backend=backend, init_method="env://")
    return rank, local_rank, world_size, device


def cleanup_distributed() -> None:
    if is_dist_available_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def reduce_mean(value: float, device: torch.device | None = None) -> float:
    """All-reduce a scalar mean across ranks (no-op when not distributed)."""
    if not is_dist_available_and_initialized() or get_world_size() == 1:
        return float(value)
    if device is None:
        device = torch.device("cuda", get_local_rank()) if torch.cuda.is_available() else torch.device("cpu")
    t = torch.tensor([float(value)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= float(get_world_size())
    return float(t.item())


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast a picklable Python object from src. Returns the object on every rank."""
    if not is_dist_available_and_initialized() or get_world_size() == 1:
        return obj
    payload = [obj if get_rank() == src else None]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def scale_num_workers(num_workers: int, world_size: int) -> int:
    """Keep total dataloader workers roughly constant when launching N processes."""
    workers = int(num_workers)
    if workers <= 0 or int(world_size) <= 1:
        return max(0, workers)
    return max(1, workers // int(world_size))


def resolve_ddp_batch_and_workers(train_cfg: dict, world_size: int) -> tuple[int, int]:
    """Derive per-GPU batch and per-process workers from the single train config.

    yaml `train.batch_size` / `train.num_workers` are the global (single-GPU) values.
    DDP keeps global batch constant: per_gpu = batch_size // nproc.
    """
    world_size = max(1, int(world_size))
    global_batch = int(train_cfg.get("batch_size", 32))
    if global_batch < 1:
        raise ValueError(f"train.batch_size must be positive, got {global_batch}")
    if global_batch % world_size != 0:
        raise ValueError(
            f"train.batch_size={global_batch} must be divisible by nproc={world_size} "
            "so DDP can split it automatically. Change batch_size or GPU count."
        )
    per_gpu_batch = global_batch // world_size
    workers = scale_num_workers(int(train_cfg.get("num_workers", 0)), world_size)
    return per_gpu_batch, workers


def print_rank0(*args: Any, **kwargs: Any) -> None:
    if is_main_process():
        print(*args, **kwargs)
