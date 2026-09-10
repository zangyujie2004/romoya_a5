from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from datasets import ZarrDataset
from tools.normalizer import DatasetNormalizer
from trainers.dist_utils import (
    barrier,
    broadcast_object,
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    print_rank0,
    reduce_mean,
    resolve_ddp_batch_and_workers,
    unwrap_model,
)
from trainers.eval_open_loop import evaluate_open_loop
from trainers.policy_trainer import (
    build_dataset,
    build_grad_scaler,
    build_policy,
    build_train_loader,
    get_checkpoint_state,
    load_checkpoint,
    resolve_mixed_precision,
    save_checkpoint,
    train_one_epoch,
)
from utils.train_utils import cfg_get, log_hparams_to_tensorboard, set_seed, sync_fm_action_horizon_from_data


def _sync_normalizer(dataset: ZarrDataset) -> None:
    """Rank0 owns fitted stats; other ranks receive them and rebuild action cache."""
    if get_world_size() <= 1:
        return
    payload = dataset.normalizer.state_dict() if is_main_process() else None
    payload = broadcast_object(payload, src=0)
    if not is_main_process():
        dataset.set_normalizer(DatasetNormalizer.load_state_dict(payload))


def _build_sampler(dataset: ZarrDataset, train_cfg: dict) -> DistributedSampler:
    return DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=True,
        drop_last=bool(train_cfg.get("drop_last", True)),
    )


def main(cfg: dict) -> None:
    ddp_cfg = dict(cfg_get(cfg, "runtime.ddp", {}) or {})
    backend = str(ddp_cfg.get("backend", "nccl"))
    find_unused = bool(ddp_cfg.get("find_unused_parameters", False))

    rank, local_rank, world_size, device = init_distributed(backend=backend)
    seed = int(cfg.get("seed", 42))
    set_seed(seed + rank)
    print_rank0(f"[ddp] world_size={world_size} backend={backend} device={device}")

    train_cfg = cfg["train"]
    resume_path = train_cfg.get("resume_path")
    skip_fit = bool(resume_path) or (world_size > 1 and not is_main_process())
    dataset = build_dataset(
        cfg,
        fit_normalizer=False if skip_fit else None,
    )
    print_rank0(f"Train windows: {len(dataset)}")

    per_gpu_bs, workers = resolve_ddp_batch_and_workers(train_cfg, world_size)
    print_rank0(
        f"[ddp] batch_size {train_cfg.get('batch_size')} -> {per_gpu_bs}/gpu "
        f"(global={per_gpu_bs * world_size}); "
        f"num_workers {train_cfg.get('num_workers')} -> {workers}/rank"
    )

    cfg = dict(cfg)
    cfg["models"] = dict(cfg.get("models") or {})
    cfg["models"]["fm"] = sync_fm_action_horizon_from_data(
        cfg["models"].get("fm") or {},
        cfg["data"],
    )

    policy = build_policy(cfg, device, dataset)
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    print_rank0(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(train_cfg["optimizer"]["lr"]),
        weight_decay=float(train_cfg["optimizer"]["weight_decay"]),
    )

    output_cfg = cfg["output"]
    output_root = str(output_cfg.get("root_dir", "outputs"))
    run_name = output_cfg.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / str(run_name)
    ckpt_dir = run_dir / "checkpoints"

    writer = None
    if is_main_process():
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        writer = SummaryWriter(log_dir=str(run_dir))
        log_hparams_to_tensorboard(writer, cfg, str(run_dir))
        print(f"TensorBoard log dir: {run_dir}")
    barrier()

    epochs = int(train_cfg.get("epochs", 100))
    ckpt_cfg = cfg.get("checkpoint", {})
    save_every = max(1, int(ckpt_cfg.get("save_every", 100)))
    grad_clip = train_cfg.get("grad_clip")
    open_loop_every = int(train_cfg.get("open_loop_test_every", 0))
    open_loop_max_batches = max(1, int(train_cfg.get("open_loop_test_max_batches", 20)))
    max_train_batches = train_cfg.get("max_train_batches")
    if max_train_batches is not None:
        max_train_batches = max(1, int(max_train_batches))
    plot_samples = int(train_cfg.get("plot_samples", 4))
    plot_dims = str(train_cfg.get("plot_dims", "auto"))
    mixed_precision = resolve_mixed_precision(train_cfg, device)
    scaler = build_grad_scaler(mixed_precision)

    print_rank0(
        f"[train] mode=ddp epochs={epochs} "
        f"open_loop_every={open_loop_every}(epoch) save_every={save_every}(epoch) "
        f"batch_size={per_gpu_bs}/gpu nproc={world_size} "
        f"global_batch={per_gpu_bs * world_size} "
        f"mixed_precision={mixed_precision} "
        f"| TB: Step/*=global_step(per-rank), Epoch/* & OpenLoop/*=epoch"
    )

    fm_cfg = cfg["models"]["fm"]
    num_inference_steps = int(fm_cfg.get("num_inference_steps", 16))
    solver = str(fm_cfg.get("solver", "euler"))

    global_step = 0
    start_epoch = 1
    if resume_path:
        resume_state = load_checkpoint(resume_path, policy, optimizer, dataset)
        global_step = int(resume_state.get("global_step", 0))
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        policy.to(device)
        print_rank0(
            f"Resumed from {resume_path} at epoch={start_epoch}, global_step={global_step}"
        )
    else:
        _sync_normalizer(dataset)

    sampler = _build_sampler(dataset, train_cfg)
    train_loader = build_train_loader(
        cfg,
        dataset,
        sampler=sampler,
        batch_size=per_gpu_bs,
        num_workers=workers,
    )
    print_rank0(f"[ddp] per-rank batches≈{len(train_loader)}")

    if world_size > 1:
        policy = DDP(
            policy,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=find_unused,
        )

    for epoch in range(start_epoch, epochs + 1):
        sampler.set_epoch(epoch)
        train_avg, global_step = train_one_epoch(
            policy,
            train_loader,
            optimizer,
            device,
            grad_clip=grad_clip,
            global_step=global_step,
            writer=writer if is_main_process() else None,
            scaler=scaler,
            mixed_precision=mixed_precision,
            max_batches=max_train_batches,
            show_pbar=is_main_process(),
        )

        train_loss = reduce_mean(float(train_avg["loss"]), device=device)
        reduced_avg = {k: reduce_mean(float(v), device=device) for k, v in train_avg.items()}

        open_loop_metrics = None
        if open_loop_every > 0 and (epoch % open_loop_every == 0 or epoch == epochs):
            if is_main_process():
                open_loop_metrics = evaluate_open_loop(
                    unwrap_model(policy),
                    dataset,
                    dataset.normalizer,
                    device,
                    epoch=epoch,
                    seed=seed,
                    max_batches=open_loop_max_batches,
                    batch_size=per_gpu_bs,
                    plot_samples=plot_samples,
                    plot_dims=plot_dims,
                    out_dir=run_dir / "open_loop",
                    writer=writer,
                    num_inference_steps=num_inference_steps,
                    solver=solver,
                )
            barrier()

        if is_main_process():
            curr_lr = optimizer.param_groups[0]["lr"]
            assert writer is not None
            writer.add_scalar("Epoch/lr", curr_lr, epoch)
            writer.add_scalar("Epoch/train_loss", train_loss, epoch)
            for key, value in reduced_avg.items():
                if key != "loss":
                    writer.add_scalar(f"Epoch/train_{key}", value, epoch)

            message = (
                f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} "
                f"(global_bs={per_gpu_bs * world_size})"
            )
            if open_loop_metrics is not None:
                message += (
                    f", open_loop_l1={open_loop_metrics['action_l1']:.6f}"
                    f", open_loop_mse={open_loop_metrics['action_mse']:.6f}"
                )
            print(message)

            state = get_checkpoint_state(
                unwrap_model(policy),
                optimizer,
                dataset,
                epoch=epoch,
                global_step=global_step,
                cfg=cfg,
            )
            save_checkpoint(ckpt_dir / "latest.pt", state)
            if epoch % save_every == 0:
                save_checkpoint(ckpt_dir / f"epoch_{epoch:04d}.pt", state)
            writer.flush()
        barrier()

    if is_main_process() and writer is not None:
        writer.close()
        print(f"Training finished. Artifacts saved in: {run_dir}")
    cleanup_distributed()
