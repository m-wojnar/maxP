#!/usr/bin/env python3
"""maxP vision pre-training entrypoint using timm models and HF streaming data."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from timm.data import create_transform, resolve_model_data_config
from timm.loss import LabelSmoothingCrossEntropy
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.tensorboard import SummaryWriter

from maxp import Parametrization

from hf_vision_data import DATASET_CONFIGS, build_dataloaders
from maxp_timm import SCALE_CONFIGS, count_trainable_params, create_model, install_pm_wrappers
from utils import append_json, collect_alignments, collect_layer_lrs, save_checkpoint, write_json


def evaluate(
    *,
    model: nn.Module,
    loader,
    device: torch.device,
    amp_enabled: bool,
    max_steps: int | None = None,
    num_classes: int | None = None,
) -> dict[str, float]:

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_top5 = 0
    total_count = 0
    steps = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.autocast(device.type, torch.bfloat16, amp_enabled):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)

            preds = logits.argmax(dim=1)
            total_loss += float(loss.item()) * xb.size(0)
            total_correct += int((preds == yb).sum().item())
            if num_classes is not None and num_classes >= 5:
                total_top5 += int(
                    (logits.topk(5, dim=1).indices == yb.unsqueeze(1)).any(dim=1).sum().item()
                )
            total_count += int(xb.size(0))
            steps += 1

            if max_steps is not None and steps >= max_steps:
                break

    model.train()

    if total_count == 0:
        return {}

    return {
        "loss/val_loss": total_loss / total_count,
        "loss/val_top1": total_correct / total_count,
        "loss/val_top5": total_top5 / total_count,
        "loss/val_samples": float(total_count),
    }


def current_lr_prefactor(optimizer: torch.optim.Optimizer) -> float:
    for group in optimizer.param_groups:
        if group.get("layer_name") == "_other":
            return float(group["lr"])
    return float(optimizer.param_groups[0]["lr"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=str, default="vit-s")
    parser.add_argument("--method", choices=["maxP", "mup-full", "mup-no"], default="maxP")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="imagenet12k")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--val-interval", type=int, default=500)
    parser.add_argument("--val-steps", type=int, default=None)
    parser.add_argument("--alignment-warmup", type=int, default=100)
    parser.add_argument("--solve-interval", type=int, default=200)
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--lr-warmup", type=int, default=None)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--no-compile", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", type=str, default="runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        amp_enabled = True
    else:
        amp_enabled = False

    scale_cfg = SCALE_CONFIGS.get(args.scale)
    dataset_cfg = DATASET_CONFIGS.get(args.dataset)

    if args.max_steps is not None:
        total_steps = args.max_steps
    else:
        steps_per_epoch = max(1, dataset_cfg.train_samples // args.batch_size)
        total_steps = steps_per_epoch * args.epochs

    num_classes = dataset_cfg.num_classes
    model = create_model(
        scale=args.scale,
        num_classes=num_classes,
        image_size=scale_cfg.image_size,
    )
    model.set_grad_checkpointing(enable=True)
    
    install_pm_wrappers(model)
    model = model.to(device)
    model.train()

    model_data_cfg = resolve_model_data_config(model)
    train_transform = create_transform(**model_data_cfg, is_training=True)
    eval_transform = create_transform(**model_data_cfg, is_training=False)

    train_loader, val_loader, _ = build_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    sample_input = torch.randn(1, 3, scale_cfg.image_size, scale_cfg.image_size, device=device)
    alignment_mode = "full" if "full" in args.method else "no"
    dynamic = args.method == "maxP"

    param = Parametrization(
        model,
        optimizer_type="adam",
        alignment=alignment_mode,
        lr_prefactor=args.lr,
        sample_input=sample_input,
        warmup_steps=args.alignment_warmup,
        solve_interval=args.solve_interval,
        sample_size=args.sample_size,
    )
    
    optimizer = torch.optim.AdamW(
        param.param_groups,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    )
    lr_warmup = args.lr_warmup or int(0.05 * total_steps)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=total_steps - lr_warmup,
        warmup_t=lr_warmup,
        warmup_prefix=True,
    )
    
    smooth_xe_loss = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)

    if not args.no_compile:
        model = torch.compile(model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"

    run_config = vars(args).copy()
    write_json(output_dir / "config.json", run_config)

    if not args.debug:
        wandb_project = os.getenv("WANDB_PROJECT", "maxP-vision")
        wandb_run_name = os.getenv("WANDB_RUN_NAME", output_dir.name)
        wandb.init(project=wandb_project, name=wandb_run_name, config=run_config)
        
        log_dir = output_dir / "tb"
        log_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(log_dir))
    
    print("\n=== maxP vision run ===")
    print(f"  scale:            {args.scale} ({scale_cfg.model_name})")
    print(f"  params:           {count_trainable_params(model)}")
    print(f"  method:           {args.method}")
    print(f"  dataset:          {args.dataset}")
    print(f"  train_samples:    {dataset_cfg.train_samples}")
    print(f"  val_samples:      {dataset_cfg.val_samples}")
    print(f"  image_size:       {scale_cfg.image_size}")
    print(f"  num_classes:      {num_classes}")
    print(f"  batch_size:       {args.batch_size}")
    print(f"  num_workers:      {args.num_workers}")
    print(f"  prefetch_factor:  {args.prefetch_factor}")
    print(f"  device:           {device}")
    print(f"  output_dir:       {output_dir}")
    print("")


    global_step = start_epoch = total_seen = 0
    align_sample: torch.Tensor | None = None

    t0 = time.time()
    last_log_time = t0
    last_log_seen = total_seen
    stop_training = False

    for epoch in range(start_epoch, args.epochs):
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if dynamic and align_sample is None:
                align_sample = xb[:args.sample_size].detach().clone().to(device)
                param.capture_initial(align_sample)

            optimizer.zero_grad()

            with torch.autocast(device.type, torch.bfloat16, amp_enabled):
                logits = model(xb)
                loss = smooth_xe_loss(logits, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step(global_step)

            if dynamic:
                param.lr_prefactor = current_lr_prefactor(optimizer)
                param.step(align_sample, optimizer)

            global_step += 1
            total_seen += int(xb.size(0))

            if global_step % args.log_interval == 0:
                now = time.time()
                elapsed = max(now - t0, 1e-6)
                interval_time = max(now - last_log_time, 1e-6)
                interval_samples = max(total_seen - last_log_seen, 1)
                interval_sps = interval_samples / interval_time
                last_log_time = now
                last_log_seen = total_seen

                if global_step % args.val_interval == 0:
                    val_interval = evaluate(
                        model=model,
                        loader=val_loader,
                        device=device,
                        amp_enabled=amp_enabled,
                        max_steps=args.val_steps,
                        num_classes=num_classes,
                    )
                else:
                    val_interval = {}

                row: dict[str, Any] = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss/train_loss": float(loss.item()),
                    "perf/samples_seen": total_seen,
                    "perf/samples_per_sec": total_seen / elapsed,
                    "perf/samples_per_sec_interval": interval_sps,
                    "lr/lr_prefactor": current_lr_prefactor(optimizer),
                    **collect_layer_lrs(param),
                    **val_interval,
                }
                if dynamic:
                    row.update(collect_alignments(param))
                append_json(metrics_path, row)

                if not args.debug:
                    wandb.log(row, step=global_step)
                
                    for key, value in row.items():
                        tb_writer.add_scalar(key, value, global_step)

                val_loss_print = row.get("loss/val_loss")
                val_txt = f" val_loss={val_loss_print:.4f}" if isinstance(val_loss_print, float) else ""
                print(
                    f"[step {global_step}] loss={row['loss/train_loss']:.4f}{val_txt}"
                    f" sps={interval_sps:.1f}"
                )

            if args.checkpoint_interval and global_step % args.checkpoint_interval == 0:
                save_checkpoint(
                    output_dir / "checkpoint" / f"step_{global_step}.pt",
                    model=model,
                    optimizer=optimizer,
                    step=global_step,
                    epoch=epoch,
                    samples_seen=total_seen,
                    args=args,
                )

            if args.max_steps is not None and global_step >= args.max_steps:
                stop_training = True
                break

        if stop_training:
            break

    eval_metrics = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        amp_enabled=amp_enabled,
        num_classes=num_classes,
    )
    elapsed = time.time() - t0

    write_json(output_dir / "final_metrics.json", eval_metrics)
    
    if not args.debug:
        wandb.log(eval_metrics, step=global_step)
        wandb.finish()
        
        for key, value in eval_metrics.items():
            tb_writer.add_scalar(key, value, global_step)
        tb_writer.close()
    
    save_checkpoint(
        output_dir / "checkpoint" / "final.pt",
        model=model,
        optimizer=optimizer,
        step=global_step,
        epoch=max(0, args.epochs - 1),
        samples_seen=total_seen,
        args=args,
    )

    print(f"\nDone. Total training time: {elapsed:.1f} seconds.")
    print(json.dumps(eval_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
