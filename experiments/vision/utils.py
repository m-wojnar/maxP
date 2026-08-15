"""Utility helpers shared across vision experiment scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from maxp import Parametrization


def collect_layer_lrs(param: Parametrization) -> dict[str, float]:
    out: dict[str, float] = {}
    for group in param.param_groups:
        layer_name = group.get("layer_name")
        if layer_name:
            out[f"lrs/{layer_name}"] = float(group["lr"])
    return out


def collect_alignments(param: Parametrization) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, pm in param._pms:
        if pm.weight is None:
            continue
        if pm.align_z0_dW is None:
            continue
        out[f"align/z0_dW/{name}"] = float(pm.align_z0_dW)
        out[f"align/dZ_w0/{name}"] = float(pm.align_dZ_w0)
        out[f"align/dZ_dW/{name}"] = float(pm.align_dZ_dW)
    return out


def save_checkpoint(
    ckpt_path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    samples_seen: int,
    args: argparse.Namespace,
    keep_latest_k: int = 2,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    model_for_io = model._orig_mod if hasattr(model, "_orig_mod") else model
    tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    torch.save(
        {
            "model": model_for_io.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "samples_seen": samples_seen,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "args": vars(args),
        },
        tmp_path,
    )
    # Atomic rename so an interrupted save never leaves a half-written ckpt.
    tmp_path.replace(ckpt_path)
    _prune_checkpoints(ckpt_path.parent, keep_latest_k)


def _prune_checkpoints(ckpt_dir: Path, keep_latest_k: int) -> None:
    """Keep only the newest `keep_latest_k` step checkpoints (final.pt is never pruned)."""
    steps = sorted(
        ckpt_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    for stale in steps[:-keep_latest_k]:
        stale.unlink(missing_ok=True)


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """Return the highest-step checkpoint in `ckpt_dir`, or final.pt, else None."""
    if not ckpt_dir.is_dir():
        return None
    steps = sorted(
        ckpt_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if steps:
        return steps[-1]
    final = ckpt_dir / "final.pt"
    return final if final.is_file() else None


def load_checkpoint(
    ckpt_path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Restore model + optimizer + RNG in-place; return progress (step/epoch/samples_seen)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_for_io = model._orig_mod if hasattr(model, "_orig_mod") else model
    model_for_io.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if ckpt.get("rng") is not None:
        torch.set_rng_state(ckpt["rng"].cpu() if hasattr(ckpt["rng"], "cpu") else ckpt["rng"])
    if ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
        # map_location may have moved these onto the GPU; set_rng_state_all needs
        # CPU ByteTensors.
        cuda_rng = [s.cpu() if hasattr(s, "cpu") else s for s in ckpt["cuda_rng"]]
        torch.cuda.set_rng_state_all(cuda_rng)
    return {
        "step": int(ckpt["step"]),
        "epoch": int(ckpt["epoch"]),
        "samples_seen": int(ckpt["samples_seen"]),
    }


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def append_json(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
