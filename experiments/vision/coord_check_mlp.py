#!/usr/bin/env python3
"""Coord-style diagnostic for ScalableMLP.

Width axis is `hidden` — the dimension that scales with model size.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from maxp import Parametrization, diagnose_axis, plot_axis, print_axis

from maxp_timm import ScalableMLP, _install_mlp_wrappers


WIDTHS = [64, 128, 256, 512, 1024]
IMAGE_SIZE = 32
NUM_CLASSES = 1000
DEPTH = 6


def _make_model(hidden: int, parametrized: bool):
    model = ScalableMLP(
        hidden=hidden,
        depth=DEPTH,
        num_classes=NUM_CLASSES,
        dropout=0.0,
        image_size=IMAGE_SIZE,
        patch_size=4,
    )
    if not parametrized:
        return model, None
    _install_mlp_wrappers(model)
    sample_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    param = Parametrization(
        model,
        lr_prefactor=1e-3,
        optimizer_type="adam",
        alignment="full",
        sample_input=sample_input,
    )
    return model, param.param_groups


def _make_input(hidden: int) -> torch.Tensor:
    return torch.randn(8, 3, IMAGE_SIZE, IMAGE_SIZE)


def _make_train_step(model, param_groups):
    opt = (
        torch.optim.AdamW(param_groups)
        if param_groups is not None
        else torch.optim.AdamW(model.parameters(), lr=1e-3)
    )
    x_fixed = torch.randn(8, 3, IMAGE_SIZE, IMAGE_SIZE)
    y_fixed = torch.randint(0, NUM_CLASSES, (8,))

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
        description="Coord check for ScalableMLP (continuous hidden width axis)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--parametrized", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    variant = "parametrized" if args.parametrized else "plain"
    print(f"ScalableMLP coord check ({variant}) — widths={WIDTHS}")

    ops, affected, act_stats = diagnose_axis(
        make_model_fn=lambda w: _make_model(w, args.parametrized),
        make_input_fn=_make_input,
        widths=WIDTHS,
        n_steps=args.steps,
        n_seeds=args.seeds,
        train_step_fn=_make_train_step,
    )
    print_axis("hidden", ops, affected, act_stats, WIDTHS)

    if args.plot:
        out_name = f"coord_check_mlp_{variant}.png"
        plot_axis("hidden", ops, affected, act_stats, WIDTHS, out_name, plot_every=1)
        print(f"Saved {out_name}")


if __name__ == "__main__":
    main()
