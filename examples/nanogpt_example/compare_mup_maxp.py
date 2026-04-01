#!/usr/bin/env python3
"""muP vs maxP comparison with WSD schedule.

Supports Shakespeare (char-level, default) and OpenWebText (GPT-2 BPE).

Usage:
    python compare_mup_maxp.py
    python compare_mup_maxp.py --steps 5000 --d-model 256
    python compare_mup_maxp.py --dataset openwebtext --preset gpt2-debug --steps 1000
    python compare_mup_maxp.py --dataset openwebtext --preset gpt2-small --steps 5000
    # Quick smoke test:
    python compare_mup_maxp.py --d-model 128 --n-layers 4 --n-heads 4 --steps 500 --warmup 50 --decay 100 --alignment-warmup 5
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.nanogpt_example.parametrized_gpt import ParametrizedGPT
from examples.nanogpt_example.sweep import (
    RunResult,
    batch_iter,
    batch_iter_mmap,
    get_device,
    load_openwebtext,
    load_shakespeare,
    pick_best,
)
from examples.nanogpt_example.replot import plot_comparison
from maxp import Parametrization


# ── Presets ──────────────────────────────────────────────────────────────

PRESETS = {
    "gpt2-small": dict(d_model=768, n_heads=12, n_layers=12, d_ff=3072, seq_len=1024),
    "gpt2-debug": dict(d_model=256, n_heads=8, n_layers=6, d_ff=1024, seq_len=256),
}


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

def _make_batch_iter(data, seq_len, batch_size, device):
    """Create appropriate batch iterator for tensor or mmap data."""
    if isinstance(data, torch.Tensor):
        return batch_iter(data, seq_len, batch_size)
    else:
        return batch_iter_mmap(data, seq_len, batch_size, device)


def train_mup(
    d_model, n_heads, n_layers, d_ff, data, vocab_size, *,
    lr, n_steps, seq_len, batch_size, seed, warmup, decay, device=None,
) -> RunResult:
    """muP with WSD schedule (static alignment)."""
    device = device or data.device
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
    it = _make_batch_iter(data, seq_len, batch_size, device)
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
    lr, n_steps, seq_len, batch_size, seed, warmup, decay, device=None,
) -> RunResult:
    """muP with WSD schedule (no alignment assumption)."""
    device = device or data.device
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
    it = _make_batch_iter(data, seq_len, batch_size, device)
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


def _sample_from_data(data, seq_len, n_samples, device):
    """Sample sequences from tensor or mmap data, returning a tensor on device."""
    if isinstance(data, torch.Tensor):
        idx = torch.randint(0, data.shape[0] - seq_len - 1, (n_samples,))
        return torch.stack([data[i : i + seq_len] for i in idx])
    else:
        idx = torch.randint(0, len(data) - seq_len - 1, (n_samples,))
        return torch.stack([
            torch.from_numpy(data[i:i+seq_len].astype(np.int64)) for i in idx
        ]).to(device)


def train_maxp(
    d_model, n_heads, n_layers, d_ff, data, vocab_size, *,
    lr, n_steps, seq_len, batch_size, seed,
    warmup, decay, alignment_warmup, solve_interval, sample_size, c_ema,
    alignment_overrides=None, norm_mode="rms", method_name="maxP",
    device=None,
) -> RunResult:
    """maxP with WSD schedule (dynamic alignment)."""
    device = device or data.device
    torch.manual_seed(seed)
    model, sample_input = _make_model(d_model, n_heads, n_layers, d_ff, vocab_size, seq_len, device)

    param = Parametrization(
        model,
        lr_prefactor=lr,
        optimizer_type="adam",
        alignment="full",
        alignment_overrides=alignment_overrides,
        warmup_steps=alignment_warmup,
        solve_interval=solve_interval,
        sample_size=sample_size,
        c_ema=c_ema,
        norm_mode=norm_mode,
        sample_input=sample_input,
    )
    optimizer = torch.optim.AdamW(param.param_groups, lr=lr)

    # Build sample input for alignment measurement
    sample_x = _sample_from_data(data, seq_len, sample_size, device)
    param.capture_initial(sample_x)

    # Track per-layer alignment and LR
    layer_history: dict[str, list[dict]] = {}
    for name, pm in param._pms:
        if pm.weight is not None:
            layer_history[name] = []

    losses = []
    it = _make_batch_iter(data, seq_len, batch_size, device)
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
        method=method_name, lr=lr,
        losses=losses, layer_history=layer_history,
    )


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="muP vs maxP comparison with WSD schedule"
    )
    parser.add_argument("--dataset", type=str, default="shakespeare",
                        choices=["shakespeare", "openwebtext"],
                        help="Dataset to train on")
    parser.add_argument("--preset", type=str, default="none",
                        choices=["none", "gpt2-small", "gpt2-debug"],
                        help="Model preset (sets d_model, n_heads, etc.)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (default: ./data/<dataset>)")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None,
                        help="FFN hidden dim (default: 4 * d_model)")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.1],
                        help="Learning rate(s) to sweep; best is picked per method")
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

    # Apply preset defaults, then explicit overrides
    defaults = {"d_model": 512, "n_heads": 8, "n_layers": 8, "d_ff": None, "seq_len": 64}
    if args.preset != "none":
        defaults.update(PRESETS[args.preset])

    d_model = args.d_model or defaults["d_model"]
    n_heads = args.n_heads or defaults["n_heads"]
    n_layers = args.n_layers or defaults["n_layers"]
    seq_len = args.seq_len or defaults["seq_len"]
    d_ff = args.d_ff or defaults["d_ff"] or 4 * d_model

    lrs = sorted(args.lrs)

    device = get_device()
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}" + (f" (preset: {args.preset})" if args.preset != "none" else ""))
    print(f"d_model={d_model}, n_heads={n_heads}, "
          f"n_layers={n_layers}, d_ff={d_ff}, seq_len={seq_len}")
    print(f"LRs={lrs}, steps={args.steps}, batch_size={args.batch_size}")
    print(f"WSD: warmup={args.warmup}, decay={args.decay}")
    print(f"maxP: alignment_warmup={args.alignment_warmup}, "
          f"solve_interval={args.solve_interval}, sample_size={args.sample_size}, "
          f"c_ema={args.c_ema}")
    print()

    # Cache setup — keyed on all hyperparams + dataset so changing config invalidates
    cache_dir = Path(args.output).parent / ".result_cache"
    cache_hparams_base = dict(
        dataset=args.dataset,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        d_ff=d_ff, seq_len=seq_len, steps=args.steps,
        batch_size=args.batch_size, warmup=args.warmup,
        decay=args.decay, seed=args.seed,
    )

    # Load data
    if args.dataset == "openwebtext":
        data_dir = args.data_dir or "./data/openwebtext"
        print(f"Loading OpenWebText from {data_dir}...")
        data, vocab_size = load_openwebtext(data_dir)
        print(f"  {len(data):,} tokens, vocab size: {vocab_size}")
    else:
        print("Loading Shakespeare...")
        data, vocab_size, chars = load_shakespeare(device)
        print(f"  {len(data):,} chars, vocab size: {vocab_size}")

    common_base = dict(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        d_ff=d_ff, data=data, vocab_size=vocab_size,
        n_steps=args.steps, seq_len=seq_len,
        batch_size=args.batch_size, seed=args.seed,
        warmup=args.warmup, decay=args.decay,
        device=device,
    )

    def _run_or_load(method_label, cache_label, file_prefix, train_fn, extra_cache_hparams=None, extra_train_kwargs=None):
        """Run all LRs for a method, return list of RunResults."""
        runs = []
        for i, lr in enumerate(lrs):
            cache_hp = {**cache_hparams_base, "lr": lr}
            if extra_cache_hparams:
                cache_hp.update(extra_cache_hparams)
            key = _cache_key(cache_label, **cache_hp)
            path = _cache_path(cache_dir, file_prefix, key)
            result = _load_result(path)
            lr_tag = f"lr={lr}"
            if result is not None:
                print(f"  [{i+1}/{len(lrs)}] {method_label} {lr_tag} — loaded from cache")
            else:
                print(f"  [{i+1}/{len(lrs)}] {method_label} {lr_tag} — training...")
                kwargs = {**common_base, "lr": lr}
                if extra_train_kwargs:
                    kwargs.update(extra_train_kwargs)
                result = train_fn(**kwargs)
                _save_result(path, result)
            tag = "DIV" if result.diverged else f"{result.final_loss:.4f}"
            print(f"        → final_loss={tag}")
            runs.append(result)
        return runs

    # ── maxP (run first — slowest, want to cache early) ──
    maxp_extra_cache = {
        "alignment_warmup": args.alignment_warmup,
        "solve_interval": args.solve_interval,
        "sample_size": args.sample_size,
        "c_ema": args.c_ema,
    }
    maxp_extra_train = {
        "alignment_warmup": args.alignment_warmup,
        "solve_interval": args.solve_interval,
        "sample_size": args.sample_size,
        "c_ema": args.c_ema,
    }
    print("\n[1/3] maxP + WSD")
    maxp_runs = _run_or_load("maxP", "maxP", "maxP", train_maxp,
                             extra_cache_hparams=maxp_extra_cache,
                             extra_train_kwargs=maxp_extra_train)
    maxp_best = pick_best(maxp_runs)

    # ── muP (full alignment) ──
    print("\n[2/3] muP (full) + WSD")
    mup_runs = _run_or_load("muP", "muP", "muP", train_mup)
    mup_best = pick_best(mup_runs)

    # ── muP (no alignment) ──
    print("\n[3/3] muP (no-align) + WSD")
    noalign_runs = _run_or_load("muP (no-align)", "muP (no-align)", "muP_no-align",
                                train_mup_noalign,
                                extra_cache_hparams={"alignment": "no"})
    noalign_best = pick_best(noalign_runs)

    # ── Summary ──
    print(f"\n{'='*60}")
    for label, runs, best in [
        ("muP (no-align)", noalign_runs, noalign_best),
        ("muP (full)",     mup_runs,     mup_best),
        ("maxP",           maxp_runs,    maxp_best),
    ]:
        print(f"  {label}:")
        for r in runs:
            tag = "DIV" if r.diverged else f"{r.final_loss:.4f}"
            marker = "  <-- best" if r is best else ""
            print(f"    lr={r.lr}  loss={tag}{marker}")
    print(f"{'='*60}")

    if not args.no_plot:
        all_runs = {
            "muP (no-align)": noalign_runs,
            "muP": mup_runs,
            "maxP": maxp_runs,
        }
        plot_comparison(
            noalign_best, mup_best, maxp_best,
            n_steps=args.steps,
            warmup=args.warmup,
            decay=args.decay,
            filename=args.output,
            all_runs=all_runs,
        )


if __name__ == "__main__":
    main()
