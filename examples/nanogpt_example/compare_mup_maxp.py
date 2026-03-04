#!/usr/bin/env python3
"""muP vs maxP comparison on larger Shakespeare GPT with WSD schedule.

Single LR (0.1), 10k steps, d_model=512, 8 layers.
Compares static alignment (muP) vs dynamic alignment (maxP).

Usage:
    python compare_mup_maxp.py
    python compare_mup_maxp.py --steps 5000 --d-model 256
    # Quick smoke test:
    python compare_mup_maxp.py --d-model 128 --n-layers 4 --n-heads 4 --steps 500 --warmup 50 --decay 100 --alignment-warmup 5
"""

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.nanogpt_example.parametrized_gpt import ParametrizedGPT
from examples.nanogpt_example.sweep import (
    RunResult,
    _LAYER_COLORS,
    _short_name,
    batch_iter,
    get_device,
    load_shakespeare,
    smooth,
)
from maxp import Parametrization


# ── Result caching ───────────────────────────────────────────────────────

def _cache_key(method: str, **kwargs) -> str:
    """Deterministic hash of method + hyperparams."""
    d = {"method": method, **{k: v for k, v in sorted(kwargs.items())}}
    return hashlib.sha256(json.dumps(d).encode()).hexdigest()[:12]


def _cache_path(cache_dir: Path, method: str, key: str) -> Path:
    return cache_dir / f"{method}_{key}.json"


def _save_result(path: Path, result: RunResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "method": result.method,
        "lr": result.lr,
        "losses": result.losses,
        "layer_history": result.layer_history,
    }
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  Cached to {path}")


def _load_result(path: Path) -> RunResult | None:
    if not path.exists():
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return RunResult(
        method=data["method"],
        lr=data["lr"],
        losses=data["losses"],
        layer_history=data.get("layer_history", {}),
    )


# ── WSD schedule ─────────────────────────────────────────────────────────

def wsd_factor(step, total, warmup=500, decay=1000):
    """Warmup-Stable-Decay schedule factor in [0, 1]."""
    if step < warmup:
        return (step + 1) / warmup
    decay_start = total - decay
    if step < decay_start:
        return 1.0
    t = (step - decay_start) / decay
    return 0.5 * (1 + math.cos(math.pi * t))


# ── Model construction ───────────────────────────────────────────────────

def _make_model(d_model, n_heads, n_layers, d_ff, vocab_size, seq_len, device):
    model = ParametrizedGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        max_seq_len=seq_len,
    ).to(device)
    sample_input = torch.randint(0, vocab_size, (1, seq_len), device=device)
    return model, sample_input


# ── Training functions ───────────────────────────────────────────────────

def train_mup(
    d_model, n_heads, n_layers, d_ff, data, vocab_size, *,
    lr, n_steps, seq_len, batch_size, seed, warmup, decay,
) -> RunResult:
    """muP with WSD schedule (static alignment)."""
    device = data.device
    torch.manual_seed(seed)
    model, sample_input = _make_model(d_model, n_heads, n_layers, d_ff, vocab_size, seq_len, device)

    param = Parametrization(
        model,
        lr_prefactor=lr,
        optimizer_type="adam",
        alignment="full",
        sample_input=sample_input,
    )
    optimizer = torch.optim.AdamW(param.param_groups, lr=lr)

    losses = []
    it = batch_iter(data, seq_len, batch_size)
    pbar = tqdm(range(n_steps), desc="muP+WSD", leave=False, ncols=90)
    for step in pbar:
        # Apply WSD schedule
        factor = wsd_factor(step, n_steps, warmup=warmup, decay=decay)
        param.lr_prefactor = lr * factor
        param._sync_lrs(optimizer)

        xb, yb = next(it)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1))
        if not math.isfinite(loss.item()):
            losses.append(float("nan"))
            break
        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    pbar.close()
    return RunResult(method="muP", lr=lr, losses=losses)


def train_maxp(
    d_model, n_heads, n_layers, d_ff, data, vocab_size, *,
    lr, n_steps, seq_len, batch_size, seed,
    warmup, decay, alignment_warmup, solve_interval, sample_size, c_ema,
) -> RunResult:
    """maxP with WSD schedule (dynamic alignment)."""
    device = data.device
    torch.manual_seed(seed)
    model, sample_input = _make_model(d_model, n_heads, n_layers, d_ff, vocab_size, seq_len, device)

    param = Parametrization(
        model,
        lr_prefactor=lr,
        optimizer_type="adam",
        alignment="full",
        warmup_steps=alignment_warmup,
        solve_interval=solve_interval,
        sample_size=sample_size,
        c_ema=c_ema,
        sample_input=sample_input,
    )
    optimizer = torch.optim.AdamW(param.param_groups, lr=lr)

    # Build sample input for alignment measurement
    sample_x = torch.stack([
        data[i : i + seq_len]
        for i in torch.randint(0, data.shape[0] - seq_len - 1, (sample_size,))
    ])
    param.capture_initial(sample_x)

    # Track per-layer alignment and LR
    layer_history: dict[str, list[dict]] = {}
    for name, pm in param._pms:
        if pm.weight is not None:
            layer_history[name] = []

    losses = []
    it = batch_iter(data, seq_len, batch_size)
    pbar = tqdm(range(n_steps), desc="maxP+WSD", leave=False, ncols=90)
    for step in pbar:
        # Apply WSD schedule before forward pass
        factor = wsd_factor(step, n_steps, warmup=warmup, decay=decay)
        param.lr_prefactor = lr * factor

        xb, yb = next(it)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1))
        if not math.isfinite(loss.item()):
            losses.append(float("nan"))
            break
        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Dynamic alignment step (also syncs LRs with current lr_prefactor)
        param.step(sample_x, optimizer)

        for name, pm in param._pms:
            if pm.weight is not None:
                layer_history[name].append({
                    "alpha": pm.alpha,
                    "omega": pm.omega,
                    "u": pm.u,
                    "lr": next(
                        g["lr"] for g in param.param_groups
                        if g.get("layer_name") == name
                    ),
                })

    pbar.close()
    return RunResult(
        method="maxP", lr=lr,
        losses=losses, layer_history=layer_history,
    )


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_comparison(
    mup_result: RunResult,
    maxp_result: RunResult,
    n_steps: int,
    warmup: int,
    decay: int,
    filename: str = "compare_mup_maxp.png",
    window: int = 50,
):
    import matplotlib.pyplot as plt

    has_hist = bool(maxp_result.layer_history)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.35)

    ax_loss = fig.add_subplot(gs[0, 0:2])
    ax_log = fig.add_subplot(gs[0, 2])
    ax_lr = fig.add_subplot(gs[0, 3])
    ax_alpha = fig.add_subplot(gs[1, 0])
    ax_omega = fig.add_subplot(gs[1, 1])
    ax_u = fig.add_subplot(gs[1, 2])
    ax_table = fig.add_subplot(gs[1, 3])

    method_style = {
        "muP": {"color": "#1f77b4", "ls": "-"},
        "maxP": {"color": "#d62728", "ls": "--"},
    }

    # ── Loss curves ──
    offset = window // 2
    for run in [mup_result, maxp_result]:
        st = method_style[run.method]
        sm = smooth(run.losses, window)
        ax_loss.plot(run.losses, alpha=0.10, color=st["color"], linewidth=0.5)
        ax_loss.plot(
            range(offset, offset + len(sm)), sm,
            color=st["color"], linewidth=2.2, linestyle=st["ls"],
            label=run.method,
        )
        ax_log.plot(run.losses, alpha=0.10, color=st["color"], linewidth=0.5)
        ax_log.plot(
            range(offset, offset + len(sm)), sm,
            color=st["color"], linewidth=2.2, linestyle=st["ls"],
            label=run.method,
        )

    # Zoom linear loss
    all_losses = mup_result.losses + maxp_result.losses
    finite = [v for v in all_losses if math.isfinite(v)]
    if finite:
        lo, hi = np.percentile(finite, 1), np.percentile(finite, 95)
        pad = 0.10 * (hi - lo)
        ax_loss.set_ylim(max(0, lo - pad), hi + pad)

    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Train Loss")
    ax_loss.set_title("Loss: muP vs maxP (WSD schedule)")
    ax_loss.legend(fontsize="small")
    ax_loss.grid(True, alpha=0.3)

    ax_log.set_xlabel("Step")
    ax_log.set_ylabel("Train Loss (log)")
    ax_log.set_yscale("log")
    ax_log.set_title("Log scale")
    ax_log.legend(fontsize="x-small")
    ax_log.grid(True, alpha=0.3)

    # ── WSD schedule overlay on loss plot ──
    sched_steps = list(range(n_steps))
    sched_vals = [wsd_factor(s, n_steps, warmup=warmup, decay=decay) for s in sched_steps]
    ax_sched = ax_loss.twinx()
    ax_sched.plot(sched_steps, sched_vals, color="gray", ls=":", lw=1.0, alpha=0.5, label="WSD schedule")
    ax_sched.set_ylabel("Schedule factor", color="gray", fontsize=8)
    ax_sched.set_ylim(-0.05, 1.15)
    ax_sched.tick_params(axis="y", labelcolor="gray", labelsize=7)

    # ── Per-layer alignment + LR for maxP ──
    if has_hist:
        hist = maxp_result.layer_history
        names = list(hist.keys())
        colors = {n: _LAYER_COLORS[i % len(_LAYER_COLORS)] for i, n in enumerate(names)}

        for name, history in hist.items():
            steps = range(len(history))
            short = _short_name(name)
            c = colors[name]
            ax_alpha.plot(steps, [h["alpha"] for h in history], color=c, lw=1.2, label=short)
            ax_omega.plot(steps, [h["omega"] for h in history], color=c, lw=1.2, label=short)
            ax_u.plot(steps, [h["u"] for h in history], color=c, lw=1.2, label=short)
            ax_lr.plot(steps, [h["lr"] for h in history], color=c, lw=1.2, label=short)

        for ax, ref, lbl in [
            (ax_alpha, 1.0, "full=1.0"), (ax_omega, 0.5, "full=0.5"), (ax_u, 1.0, "full=1.0"),
        ]:
            ax.axhline(ref, color="k", ls=":", lw=0.8, alpha=0.5, label=lbl)

        for ax, key in [(ax_alpha, "alpha"), (ax_omega, "omega"), (ax_u, "u"), (ax_lr, "lr")]:
            all_vals = []
            for history in hist.values():
                all_vals.extend(h[key] for h in history if math.isfinite(h[key]))
            if all_vals:
                lo, hi = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
                pad = 0.15 * max(hi - lo, 1e-8)
                ax.set_ylim(lo - pad, hi + pad)

    for ax, title, ylabel in [
        (ax_alpha, r"$\alpha$ (z₀ @ $\Delta$w)", r"$\alpha$"),
        (ax_omega, r"$\omega$ ($\Delta$z @ w₀)", r"$\omega$"),
        (ax_u, r"$u$ ($\Delta$z @ $\Delta$w)", r"$u$"),
        (ax_lr, "Per-layer LR (maxP)", "LR"),
    ]:
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize="x-small", ncol=2)
        ax.grid(True, alpha=0.3)

    # ── Summary table ──
    ax_table.axis("off")
    rows = []
    for run in [mup_result, maxp_result]:
        tag = f"{run.final_loss:.4f}" if not run.diverged else "DIV"
        rows.append([run.method, f"{run.lr}", tag])
    table = ax_table.table(
        cellText=rows,
        colLabels=["Method", "LR", "Final Loss"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax_table.set_title("Results", fontsize=10)

    fig.suptitle("muP vs maxP (WSD schedule) — Shakespeare GPT", fontsize=14)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {filename}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="muP vs maxP comparison with WSD schedule on Shakespeare GPT"
    )
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=None,
                        help="FFN hidden dim (default: 4 * d_model)")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=500,
                        help="WSD warmup steps")
    parser.add_argument("--decay", type=int, default=1000,
                        help="WSD decay steps (at end of training)")
    parser.add_argument("--alignment-warmup", type=int, default=10,
                        help="Steps before first alignment LP re-solve")
    parser.add_argument("--solve-interval", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--c-ema", type=float, default=0.0,
                        help="EMA smoothing for c values (0=instant, 0.99=very slow)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output", type=str, default="compare_mup_maxp.png")
    args = parser.parse_args()

    d_ff = args.d_ff or 4 * args.d_model

    device = get_device()
    print(f"Device: {device}")
    print(f"d_model={args.d_model}, n_heads={args.n_heads}, "
          f"n_layers={args.n_layers}, d_ff={d_ff}, seq_len={args.seq_len}")
    print(f"LR={args.lr}, steps={args.steps}, batch_size={args.batch_size}")
    print(f"WSD: warmup={args.warmup}, decay={args.decay}")
    print(f"maxP: alignment_warmup={args.alignment_warmup}, "
          f"solve_interval={args.solve_interval}, sample_size={args.sample_size}, "
          f"c_ema={args.c_ema}")
    print()

    # Cache setup — keyed on all hyperparams so changing config invalidates
    cache_dir = Path(args.output).parent / ".result_cache"
    cache_hparams = dict(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=d_ff, seq_len=args.seq_len, steps=args.steps,
        batch_size=args.batch_size, lr=args.lr, warmup=args.warmup,
        decay=args.decay, seed=args.seed,
    )

    print("Loading Shakespeare...")
    data, vocab_size, chars = load_shakespeare(device)
    print(f"  {len(data):,} chars, vocab size: {vocab_size}")

    common = dict(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=d_ff, data=data, vocab_size=vocab_size,
        lr=args.lr, n_steps=args.steps, seq_len=args.seq_len,
        batch_size=args.batch_size, seed=args.seed,
        warmup=args.warmup, decay=args.decay,
    )

    # ── maxP (run first — slower, want to cache early) ──
    maxp_hparams = {**cache_hparams,
                    "alignment_warmup": args.alignment_warmup,
                    "solve_interval": args.solve_interval,
                    "sample_size": args.sample_size,
                    "c_ema": args.c_ema}
    maxp_key = _cache_key("maxP", **maxp_hparams)
    maxp_cache = _cache_path(cache_dir, "maxP", maxp_key)
    maxp_result = _load_result(maxp_cache)
    if maxp_result is not None:
        print(f"\n[1/2] maxP + WSD — loaded from cache")
    else:
        print(f"\n[1/2] Training maxP + WSD...")
        maxp_result = train_maxp(
            **common,
            alignment_warmup=args.alignment_warmup,
            solve_interval=args.solve_interval,
            sample_size=args.sample_size,
            c_ema=args.c_ema,
        )
        _save_result(maxp_cache, maxp_result)
    maxp_tag = "DIV" if maxp_result.diverged else f"{maxp_result.final_loss:.4f}"
    print(f"  → maxP final_loss={maxp_tag}")

    # ── muP ──
    mup_key = _cache_key("muP", **cache_hparams)
    mup_cache = _cache_path(cache_dir, "muP", mup_key)
    mup_result = _load_result(mup_cache)
    if mup_result is not None:
        print(f"\n[2/2] muP + WSD — loaded from cache")
    else:
        print(f"\n[2/2] Training muP + WSD...")
        mup_result = train_mup(**common)
        _save_result(mup_cache, mup_result)
    mup_tag = "DIV" if mup_result.diverged else f"{mup_result.final_loss:.4f}"
    print(f"  → muP final_loss={mup_tag}")

    print(f"\n{'='*50}")
    print(f"  muP  loss={mup_tag}")
    print(f"  maxP loss={maxp_tag}")
    print(f"{'='*50}")

    if not args.no_plot:
        plot_comparison(
            mup_result, maxp_result,
            n_steps=args.steps,
            warmup=args.warmup,
            decay=args.decay,
            filename=args.output,
        )


if __name__ == "__main__":
    main()
