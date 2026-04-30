#!/usr/bin/env python3
"""Coord-style diagnostic for a custom width-scalable ConvNeXt model.

Uses a hand-rolled ConvNeXt-V1 style model with a single base width `w`
(channel progression [w, 2w, 4w, 8w]) so the scaling axis is continuous
and interpretable — unlike discrete timm variants (T/S/B/L).

Standard PyTorch init; maxP can reinitialize weights freely.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from maxp import Parametrization, ParametrizedModule, diagnose_axis, plot_axis, print_axis


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _LayerNorm2d(nn.Module):
    """NCHW wrapper around nn.LayerNorm (no learnable params in the conv path)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim, bias=True)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + x


class ScalableConvNeXt(nn.Module):
    """ConvNeXt-V1 with configurable base width w.

    Channel progression: [w, 2w, 4w, 8w].
    No layer scale, no GRN — keeps init simple for maxP.
    """

    def __init__(
        self,
        w: int,
        num_classes: int = 1000,
        depths: tuple[int, ...] = (2, 2, 6, 2),
    ) -> None:
        super().__init__()
        dims = [w, 2 * w, 4 * w, 8 * w]

        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4, bias=True),
            _LayerNorm2d(dims[0]),
        )

        self.stages = nn.ModuleList([
            nn.Sequential(*[_ConvNeXtBlock(d) for _ in range(depth)])
            for d, depth in zip(dims, depths)
        ])

        # Between-stage downsampling (3 transitions for 4 stages)
        self.downsamples = nn.ModuleList([
            nn.Sequential(
                _LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2, bias=True),
            )
            for i in range(3)
        ])

        self.head_norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        x = x.mean([-2, -1])
        x = self.head_norm(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# ParametrizedModule installation
# ---------------------------------------------------------------------------

def _conv_fan_in(m: nn.Conv2d) -> int:
    # Depthwise: scaling axis is channel count, not spatial fan-in.
    if m.groups == m.in_channels == m.out_channels:
        return m.in_channels
    return (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]


def install_wrappers(model: ScalableConvNeXt) -> list[str]:
    wrapped: list[str] = []

    # Stem Conv2d — embedding
    stem_conv = model.stem[0]
    model.stem[0] = ParametrizedModule(stem_conv, width_dim=_conv_fan_in(stem_conv), layer_type="embedding")
    wrapped.append("stem.0")

    # Stage blocks: dwconv, pwconv1, pwconv2
    for si, stage in enumerate(model.stages):
        for bi, block in enumerate(stage):
            base = f"stages.{si}.{bi}"

            dw = block.dwconv
            block.dwconv = ParametrizedModule(dw, width_dim=_conv_fan_in(dw), layer_type="hidden")
            wrapped.append(f"{base}.dwconv")

            pw1 = block.pwconv1
            block.pwconv1 = ParametrizedModule(pw1, width_dim=pw1.in_features, layer_type="hidden")
            wrapped.append(f"{base}.pwconv1")

            pw2 = block.pwconv2
            block.pwconv2 = ParametrizedModule(pw2, width_dim=pw2.in_features, layer_type="hidden")
            wrapped.append(f"{base}.pwconv2")

    # Downsample Conv2d — hidden
    for di, ds in enumerate(model.downsamples):
        conv = ds[1]
        model.downsamples[di][1] = ParametrizedModule(conv, width_dim=_conv_fan_in(conv), layer_type="hidden")
        wrapped.append(f"downsamples.{di}.1")

    # Head Linear — readout
    head = model.head
    model.head = ParametrizedModule(head, width_dim=head.in_features, layer_type="readout")
    wrapped.append("head")

    return wrapped


# ---------------------------------------------------------------------------
# Coord check harness
# ---------------------------------------------------------------------------

WIDTHS = [32, 48, 64, 96, 128]  # base channel widths → dims [w, 2w, 4w, 8w]


def _make_model(w: int, parametrized: bool):
    model = ScalableConvNeXt(w=w, num_classes=1000)
    if not parametrized:
        return model, None
    install_wrappers(model)
    param = Parametrization(
        model,
        lr_prefactor=1e-3,
        optimizer_type="adam",
        alignment="full",
        sample_input=None,
    )
    return model, param.param_groups


def _make_input(w: int) -> torch.Tensor:
    return torch.randn(2, 3, 224, 224)


def _make_train_step(model, param_groups):
    opt = (
        torch.optim.AdamW(param_groups)
        if param_groups is not None
        else torch.optim.AdamW(model.parameters(), lr=1e-3)
    )
    x_fixed = torch.randn(2, 3, 224, 224)
    y_fixed = torch.randint(0, 1000, (2,))

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
        description="Coord check for scalable ConvNeXt (continuous width axis)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--parametrized", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    variant = "parametrized" if args.parametrized else "plain"
    print(f"Scalable ConvNeXt coord check ({variant}) — widths={WIDTHS}")

    ops, affected, act_stats = diagnose_axis(
        make_model_fn=lambda w: _make_model(w, args.parametrized),
        make_input_fn=_make_input,
        widths=WIDTHS,
        n_steps=args.steps,
        n_seeds=args.seeds,
        train_step_fn=_make_train_step,
    )
    print_axis("base_width", ops, affected, act_stats, WIDTHS)

    if args.plot:
        out_name = f"coord_check_convnext_scalable_{variant}.png"
        plot_axis("base_width", ops, affected, act_stats, WIDTHS, out_name, plot_every=1)
        print(f"Saved {out_name}")


if __name__ == "__main__":
    main()
