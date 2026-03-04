#!/usr/bin/env python3
"""muP vs maxP comparison on larger Shakespeare GPT with WSD schedule.

Single LR (0.1), 10k steps, d_model=512, 8 layers.
Compares muP (no-alignment) vs muP (full-alignment) vs maxP (dynamic).

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
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.nanogpt_example.parametrized_gpt import ParametrizedGPT
from examples.nanogpt_example.sweep import (
    RunResult,
    batch_iter,
    get_device,
    load_shakespeare,
)
from examples.nanogpt_example.replot import plot_comparison
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


def train_mup_noalign(
    d_model, n_heads, n_layers, d_ff, data, vocab_size, *,
    lr, n_steps, seq_len, batch_size, seed, warmup, decay,
) -> RunResult:
    """muP with WSD schedule (no alignment assumption)."""
    device = data.device
    torch.manual_seed(seed)
    model, sample_input = _make_model(d_model, n_heads, n_layers, d_ff, vocab_size, seq_len, device)

    param = Parametrization(
        model,
        lr_prefactor=lr,
        optimizer_type="adam",
        alignment="no",
        sample_input=sample_input,
    )
    optimizer = torch.optim.AdamW(param.param_groups, lr=lr)

    losses = []
    it = batch_iter(data, seq_len, batch_size)
    pbar = tqdm(range(n_steps), desc="muP(no-align)+WSD", leave=False, ncols=90)
    for step in pbar:
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
    return RunResult(method="muP (no-align)", lr=lr, losses=losses)


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

    # ── maxP (run first — slowest, want to cache early) ──
    maxp_hparams = {**cache_hparams,
                    "alignment_warmup": args.alignment_warmup,
                    "solve_interval": args.solve_interval,
                    "sample_size": args.sample_size,
                    "c_ema": args.c_ema}
    maxp_key = _cache_key("maxP", **maxp_hparams)
    maxp_cache = _cache_path(cache_dir, "maxP", maxp_key)
    maxp_result = _load_result(maxp_cache)
    if maxp_result is not None:
        print("\n[1/3] maxP + WSD — loaded from cache")
    else:
        print("\n[1/3] Training maxP + WSD...")
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

    # ── muP (full alignment) ──
    mup_key = _cache_key("muP", **cache_hparams)
    mup_cache = _cache_path(cache_dir, "muP", mup_key)
    mup_result = _load_result(mup_cache)
    if mup_result is not None:
        print("\n[2/3] muP (full) + WSD — loaded from cache")
    else:
        print("\n[2/3] Training muP (full) + WSD...")
        mup_result = train_mup(**common)
        _save_result(mup_cache, mup_result)
    mup_tag = "DIV" if mup_result.diverged else f"{mup_result.final_loss:.4f}"
    print(f"  → muP final_loss={mup_tag}")

    # ── muP (no alignment) ──
    noalign_hparams = {**cache_hparams, "alignment": "no"}
    noalign_key = _cache_key("muP (no-align)", **noalign_hparams)
    noalign_cache = _cache_path(cache_dir, "muP_no-align", noalign_key)
    noalign_result = _load_result(noalign_cache)
    if noalign_result is not None:
        print("\n[3/3] muP (no-align) + WSD — loaded from cache")
    else:
        print("\n[3/3] Training muP (no-align) + WSD...")
        noalign_result = train_mup_noalign(**common)
        _save_result(noalign_cache, noalign_result)
    noalign_tag = "DIV" if noalign_result.diverged else f"{noalign_result.final_loss:.4f}"
    print(f"  → muP (no-align) final_loss={noalign_tag}")

    print(f"\n{'='*50}")
    print(f"  muP (no-align) loss={noalign_tag}")
    print(f"  muP (full)     loss={mup_tag}")
    print(f"  maxP           loss={maxp_tag}")
    print(f"{'='*50}")

    if not args.no_plot:
        plot_comparison(
            noalign_result, mup_result, maxp_result,
            n_steps=args.steps,
            warmup=args.warmup,
            decay=args.decay,
            filename=args.output,
        )


if __name__ == "__main__":
    main()
