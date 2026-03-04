"""Run diagnosis + coord check for the nanoGPT model.

Usage:
    python examples/nanogpt_example/coord_check.py --plot                 # vanilla
    python examples/nanogpt_example/coord_check.py --parametrized --plot  # muP
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from maxp import Parametrization, classify, diagnose_axis, print_axis, plot_axis
from maxp.diagnose import op_label
from maxp.trace import ClassifiedOp

from examples.nanogpt_example.gpt import GPT
from examples.nanogpt_example.parametrized_gpt import (
    ParametrizedGPT, make_parametrized_gpt,
)


# --- Model factories ---

def make_gpt(**kwargs):
    defaults = dict(vocab_size=256, d_model=128, n_heads=4, d_ff=512, n_layers=2)
    defaults.update(kwargs)
    return GPT(**defaults), None


def make_pgpt(**kwargs):
    """Create a ParametrizedModule-wrapped GPT with ABC parametrization."""
    defaults = dict(vocab_size=256, d_model=128, n_heads=4, d_ff=512, n_layers=2)
    defaults.update(kwargs)
    model, param = make_parametrized_gpt(**defaults)
    return model, param.param_groups


# --- Axis configs ---

AXES = {
    "d_model": {
        "make_model": lambda w: make_gpt(d_model=w, n_heads=w // 16, d_ff=4 * w),
        "widths": [64, 128, 256, 512, 1024],
    },
    "head_dim": {
        "make_model": lambda w: make_gpt(d_model=4 * w, n_heads=4, d_ff=16 * w),
        "widths": [16, 32, 64, 128, 256],
    },
}

AXES_PARAMETRIZED = {
    "d_model": {
        "make_model": lambda w: make_pgpt(d_model=w, n_heads=w // 16, d_ff=4 * w),
        "widths": [64, 128, 256, 512, 1024],
    },
    "head_dim": {
        "make_model": lambda w: make_pgpt(d_model=4 * w, n_heads=4, d_ff=16 * w),
        "widths": [16, 32, 64, 128, 256],
    },
}


def make_input(width):
    return torch.randint(0, 256, (1, 8))


def _op_key(op: ClassifiedOp) -> str:
    """Stable identity for an op across axes."""
    if op.param_name:
        return op.param_name
    return f"{op.module_path}:{op.source_loc}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--parametrized", action="store_true",
                        help="Use ParametrizedModule GPT with ABC parametrization")
    parser.add_argument("--plot-every", type=int, default=1,
                        help="Plot every N-th step (default: every step)")
    args = parser.parse_args()

    axes = AXES_PARAMETRIZED if args.parametrized else AXES
    variant = "parametrized" if args.parametrized else "vanilla"
    print(f"Using {variant} GPT")

    claimed: dict[str, str] = {}
    shown_embeddings: set[str] = set()

    for axis_name, axis_cfg in axes.items():
        print(f"\nDiagnosing axis: {axis_name}...")
        ops, affected, act_stats = diagnose_axis(
            make_model_fn=axis_cfg["make_model"],
            make_input_fn=make_input,
            widths=axis_cfg["widths"],
            n_steps=args.steps,
            n_seeds=args.seeds,
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
                    print(f"  WARNING: {op_label(op)} already in '{claimed[key]}' axis, skipping")
                else:
                    claimed[key] = axis_name
                    deduped.append(i)

        print_axis(axis_name, ops, deduped, act_stats, axis_cfg["widths"])

        if args.plot:
            plot_axis(axis_name, ops, deduped, act_stats,
                      axis_cfg["widths"], f"diagnose_{variant}_{axis_name}.png",
                      plot_every=args.plot_every)
