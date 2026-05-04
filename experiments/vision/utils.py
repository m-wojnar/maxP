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


def collect_alignments(param: Parametrization) -> dict[str, dict[str, float]]:
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
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    model_for_io = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "model": model_for_io.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "samples_seen": samples_seen,
            "args": vars(args),
        },
        ckpt_path,
    )


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def append_json(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
