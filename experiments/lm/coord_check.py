"""Coord check for the parametrized LLaMA-3 model.

Verifies that activation norms are O(1) in width under ABC parametrization.

Usage:
    python experiments/lm/coord_check.py                   # plain (no PM)
    python experiments/lm/coord_check.py --parametrized    # maxP
    python experiments/lm/coord_check.py --parametrized --plot
"""

from __future__ import annotations

import argparse

import torch

from maxp import Parametrization, diagnose_axis, print_axis, plot_axis
from maxp.diagnose import op_label
from maxp.trace import ClassifiedOp

from maxp_converter import install_pm_wrappers
from maxp_llama3 import _make_model_config


VOCAB_SIZE = 2048
N_LAYERS = 2


def _make_llama3(dim: int, n_heads: int, n_kv_heads: int, parametrized: bool):
    cfg = _make_model_config(
        dim=dim, n_layers=N_LAYERS, n_heads=n_heads,
        n_kv_heads=n_kv_heads, vocab_size=VOCAB_SIZE,
    )
    model = cfg.build()
    if not parametrized:
        return model, None
    install_pm_wrappers(model)
    sample = torch.randint(0, VOCAB_SIZE, (1, 16))
    param = Parametrization(model, sample_input=sample, lr_prefactor=1.0)
    return model, param.param_groups


def _make_input(width: int) -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (1, 8))


def _make_train_step(model, param_groups):
    """One AdamW step for LLaMA-3 (uses tok_embeddings, not tok_emb)."""
    import torch.nn.functional as F

    if param_groups is not None:
        opt = torch.optim.AdamW(param_groups)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    x_fixed = torch.randint(0, VOCAB_SIZE, (4, 16))
    targets_fixed = torch.randint(0, VOCAB_SIZE, (4, 16))

    def step(model, step_idx):
        logits = model(x_fixed)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets_fixed.reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad()

    return step


def _op_key(op: ClassifiedOp) -> str:
    if op.param_name:
        return op.param_name
    return f"{op.module_path}:{op.source_loc}"


# ---------------------------------------------------------------------------
# Axis configs
# ---------------------------------------------------------------------------

# d_model axis: head_dim fixed at 64, scale dim and n_heads together
D_MODEL_WIDTHS = [128, 256, 512, 768, 1024, 1536, 2048]


def _make_d_model(width: int, parametrized: bool):
    n_heads = width // 64
    return _make_llama3(width, n_heads=n_heads, n_kv_heads=n_heads // 2,
                        parametrized=parametrized)


# head_dim axis: n_heads=8, n_kv_heads=4 fixed; dim = 8 * head_dim
HEAD_DIM_WIDTHS = [32, 64, 128, 256]


def _make_head_dim(width: int, parametrized: bool):
    return _make_llama3(dim=8 * width, n_heads=8, n_kv_heads=4,
                        parametrized=parametrized)


AXES = {
    "d_model": {
        "make_model": _make_d_model,
        "widths": D_MODEL_WIDTHS,
    },
    "head_dim": {
        "make_model": _make_head_dim,
        "widths": HEAD_DIM_WIDTHS,
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Coord check for parametrized LLaMA-3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--parametrized", action="store_true",
                        help="Use maxP ABC parametrization")
    parser.add_argument("--plot", action="store_true",
                        help="Save PNG plots")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--plot-every", type=int, default=1)
    args = parser.parse_args()

    variant = "parametrized" if args.parametrized else "plain"
    print(f"LLaMA-3 coord check — {variant}")

    claimed: dict[str, str] = {}
    shown_embeddings: set[str] = set()

    for axis_name, axis_cfg in AXES.items():
        print(f"\nDiagnosing axis: {axis_name} ...")
        ops, affected, act_stats = diagnose_axis(
            make_model_fn=lambda w, cfg=axis_cfg: cfg["make_model"](w, args.parametrized),
            make_input_fn=_make_input,
            widths=axis_cfg["widths"],
            n_steps=args.steps,
            n_seeds=args.seeds,
            train_step_fn=_make_train_step,
        )

        deduped = []
        for i in affected:
            op = ops[i]
            key = _op_key(op)

            if op.layer_type == "embedding" and op.op != "embedding":
                continue
            elif op.layer_type == "embedding" and op.op == "embedding":
                if key not in shown_embeddings:
                    shown_embeddings.add(key)
                    deduped.append(i)
            else:
                if key in claimed:
                    print(f"  NOTE: {op_label(op)} already in '{claimed[key]}' axis, skipping")
                else:
                    claimed[key] = axis_name
                    deduped.append(i)

        print_axis(axis_name, ops, deduped, act_stats, axis_cfg["widths"])

        if args.plot:
            fname = f"coord_check_{variant}_{axis_name}.png"
            plot_axis(axis_name, ops, deduped, act_stats,
                      axis_cfg["widths"], fname, plot_every=args.plot_every)
            print(f"  Saved {fname}")
