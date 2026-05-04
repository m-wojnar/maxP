#!/usr/bin/env python3
"""Coord-style diagnostic for wrapped timm ViT models."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from maxp import Parametrization, diagnose_axis, plot_axis, print_axis

from maxp_timm import SCALE_CONFIGS, create_model, install_pm_wrappers


WIDTH_TO_SCALE = {
    192: "debug",   # vit_tiny
    384: "vit-s",   # vit_small
    768: "vit-b",   # vit_base
}


def _make_model(width: int, parametrized: bool):
    scale = WIDTH_TO_SCALE[width]
    cfg = SCALE_CONFIGS[scale]
    model = create_model(
        scale=scale,
        num_classes=1000,
        image_size=cfg.image_size,
    )
    if not parametrized:
        return model, None
    install_pm_wrappers(model)
    sample_input = torch.randn(1, 3, cfg.image_size, cfg.image_size)
    param = Parametrization(
        model,
        lr_prefactor=1e-3,
        optimizer_type="adam",
        alignment="full",
        sample_input=sample_input,
    )
    return model, param.param_groups


def _make_input(width: int) -> torch.Tensor:
    scale = WIDTH_TO_SCALE[width]
    cfg = SCALE_CONFIGS[scale]
    return torch.randn(32, 3, cfg.image_size, cfg.image_size)


def _make_train_step(model, param_groups):
    if param_groups is not None:
        opt = torch.optim.AdamW(param_groups)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    x_fixed = torch.randn(32, 3, 224, 224)
    y_fixed = torch.randint(0, 1000, (32,))

    def step(model, step_idx):
        del step_idx
        logits = model(x_fixed)
        loss = F.cross_entropy(logits, y_fixed)
        loss.backward()
        opt.step()
        opt.zero_grad()

    return step


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coord-style diagnostic for timm ViT wrappers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--parametrized", action="store_true")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    widths = sorted(WIDTH_TO_SCALE)
    variant = "parametrized" if args.parametrized else "plain"
    print(f"ViT coord check ({variant}) — widths={widths}")

    ops, affected, act_stats = diagnose_axis(
        make_model_fn=lambda w: _make_model(w, args.parametrized),
        make_input_fn=_make_input,
        widths=widths,
        n_steps=args.steps,
        n_seeds=args.seeds,
        train_step_fn=_make_train_step,
    )
    print_axis("embed_dim", ops, affected, act_stats, widths)

    if args.plot:
        out_name = f"coord_check_vit_{variant}.png"
        plot_axis(
            "embed_dim",
            ops,
            affected,
            act_stats,
            widths,
            out_name,
            plot_every=1,
        )
        print(f"Saved {out_name}")


if __name__ == "__main__":
    main()
