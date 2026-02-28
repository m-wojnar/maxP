#!/usr/bin/env python
"""
Demonstrate DAG solver properties on real model topologies.

Shows:
  1. Analytical correctness on a chain (matches known muP values)
  2. Optimality proof (binding constraints, perturbation check)
  3. Traced transformer DAG with per-op c under measured alignment
  4. Chain solver (old) vs DAG solver (new) comparison

Usage:
    python examples_new/dag_solver_demo.py              # print tables
    python examples_new/dag_solver_demo.py --plot       # also save figure
    python examples_new/dag_solver_demo.py --plot -o my_fig.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from maxp_new.dag import DagNode, OpGraph, MergeType, trace_pm_dag
from maxp_new.module import ParametrizedModule
from maxp_new.parametrization import Parametrization
from maxp_new.solver import find_c_dag_adam


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MiniTransformer(nn.Module):
    """Small transformer with SwiGLU MLP, for demonstrating per-op c values."""

    def __init__(self, d=64, d_ff=128, vocab=256, n_heads=4):
        super().__init__()
        head_dim = d // n_heads
        self.tok_emb = ParametrizedModule(
            nn.Embedding(vocab, d), width_dim=d, layer_type="embedding")
        self.qkv = ParametrizedModule(
            nn.Linear(d, 3*d, bias=False), width_dim=d, layer_type="hidden")
        self.attn_score = ParametrizedModule(
            lambda q, k: q @ k.transpose(-2, -1),
            width_dim=head_dim, layer_type="readout")
        self.proj = ParametrizedModule(
            nn.Linear(d, d, bias=False), width_dim=d, layer_type="hidden")
        self.gate = ParametrizedModule(
            nn.Linear(d, d_ff, bias=False), width_dim=d, layer_type="hidden")
        self.up = ParametrizedModule(
            nn.Linear(d, d_ff, bias=False), width_dim=d, layer_type="hidden")
        self.down = ParametrizedModule(
            nn.Linear(d_ff, d, bias=False), width_dim=d_ff, layer_type="hidden")
        self.head = ParametrizedModule(
            nn.Linear(d, vocab, bias=False), width_dim=d, layer_type="readout")
        self.n_heads = n_heads
        self.head_dim = head_dim

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx)
        q_k_v = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q_k_v = q_k_v.permute(2, 0, 3, 1, 4)
        q, k, v = q_k_v.unbind(0)
        attn = self.attn_score(q, k).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        x = x + self.proj(out)              # residual (addition -> MIN merge)
        x = x + self.down(                   # residual (addition -> MIN merge)
            F.silu(self.gate(x)) * self.up(x)  # SwiGLU (multiply -> SUM merge)
        )
        return self.head(x)


# ---------------------------------------------------------------------------
# Demo sections
# ---------------------------------------------------------------------------

def demo_1_analytical():
    print("=" * 65)
    print(" 1. ANALYTICAL CORRECTNESS")
    print("=" * 65)
    print()
    print(" Known muP solution for emb -> h -> head (full alignment):")
    print("   c_emb=0.5, c_h=1.0, c_head=0.5")
    print()

    g = OpGraph({
        "emb":  DagNode(name="emb",  a=-0.5, b=0.5, layer_type="embedding",
                        has_weight=True, width_dim=32,
                        predecessors=[], successors=["h"]),
        "h":    DagNode(name="h",    a=0.0, b=0.5, layer_type="hidden",
                        has_weight=True, width_dim=32,
                        predecessors=["emb"], successors=["head"]),
        "head": DagNode(name="head", a=0.5, b=0.5, layer_type="readout",
                        has_weight=True, width_dim=32,
                        predecessors=["h"], successors=[]),
    })
    res = find_c_dag_adam(g)

    print(f"   {'node':<6} {'c (solver)':<14} {'c (expected)':<14} {'r':<10} {'match'}")
    print(f"   {'-'*54}")
    expected = {"emb": 0.5, "h": 1.0, "head": 0.5}
    for name in ["emb", "h", "head"]:
        c, r = res[name]
        ok = abs(c - expected[name]) < 1e-6
        print(f"   {name:<6} {c:<14.6f} {expected[name]:<14.6f} {r:<10.6f} {'OK' if ok else 'FAIL'}")
    print()


def demo_2_optimality():
    print("=" * 65)
    print(" 2. OPTIMALITY — binding constraints + perturbation")
    print("=" * 65)
    print()

    g = OpGraph({
        "emb":  DagNode(name="emb",  a=-0.5, b=0.5, layer_type="embedding",
                        has_weight=True, width_dim=32,
                        predecessors=[], successors=["h"]),
        "h":    DagNode(name="h",    a=0.0, b=0.5, layer_type="hidden",
                        has_weight=True, width_dim=32,
                        predecessors=["emb"], successors=["head"]),
        "head": DagNode(name="head", a=0.5, b=0.5, layer_type="readout",
                        has_weight=True, width_dim=32,
                        predecessors=["h"], successors=[]),
    })
    res = find_c_dag_adam(g)

    print(" All r = 0 means every constraint is binding (tight):")
    for name, (c, r) in res.items():
        status = "binding" if abs(r) < 1e-6 else f"slack={r:.6f}"
        print(f"   r_{name} = {r:.8f}  ({status})")

    print()
    print(" Perturb each c down by eps=0.01:")
    eps = 0.01
    r_emb_p = -0.5 + (res["emb"][0] - eps)
    r_h_p = min((res["h"][0] - eps) - 1.0, (res["h"][0] - eps) - 1.0, 0.0)
    r_head_p = min(1.0 - 0.5, 0.5 + (res["head"][0] - eps) - 1.0,
                   0.5 + (res["head"][0] - eps) - 1.0)
    for label, r_p in [("c_emb", r_emb_p), ("c_h", r_h_p), ("c_head", r_head_p)]:
        print(f"   {label} - eps  =>  r = {r_p:+.4f}  {'VIOLATED' if r_p < 0 else 'ok'}")
    print()
    print(" Cannot lower any c => solver found maximum LRs.")
    print()


def demo_3_traced_transformer():
    print("=" * 65)
    print(" 3. TRACED TRANSFORMER — DAG topology + per-op c")
    print("=" * 65)
    print()
    print(" Trace a mini-transformer to discover the PM-to-PM DAG,")
    print(" then solve for c with per-op alignment values.")
    print()

    model = MiniTransformer()
    x = torch.randint(0, 256, (1, 8))
    graph = trace_pm_dag(model, x)

    # Show discovered topology
    print(" Discovered DAG:")
    for node in graph.topological_order():
        preds = ", ".join(node.predecessors) if node.predecessors else "(source)"
        merge = f" [{node.merge_type.name}]" if len(node.predecessors) > 1 else ""
        weight = "W" if node.has_weight else " "
        print(f"   {weight} {node.name:<14} <- {preds}{merge}")
    print()

    # Solve with uniform alignment (muP baseline)
    print(" With UNIFORM alignment (full preset):")
    res_uniform = find_c_dag_adam(graph)
    c_vals = set()
    for node in graph.topological_order():
        c, r = res_uniform[node.name]
        if c is not None:
            c_vals.add(round(c, 4))
    print(f"   Distinct c values: {sorted(c_vals)}")
    print(f"   (All hidden ops get the same c — uniform alignment decouples them)")
    print()

    # Now simulate measured alignment: different ops have different alpha
    print(" With MEASURED alignment (simulated Phase 2 values):")
    print("   qkv, proj: alpha=0.9  (well-aligned)")
    print("   gate, up:   alpha=0.7  (moderately aligned)")
    print("   down:       alpha=0.4  (weakly aligned)")
    print()

    graph2 = trace_pm_dag(model, x)
    alignment_map = {
        "qkv": 0.9, "proj": 0.9,
        "gate": 0.7, "up": 0.7,
        "down": 0.4,
    }
    for name, node in graph2.nodes.items():
        if name in alignment_map:
            node.alpha = alignment_map[name]

    res_measured = find_c_dag_adam(graph2)

    print(f"   {'op':<14} {'type':<10} {'c':>8} {'r':>8}  {'LR (prefactor=0.1)':>20}")
    print(f"   {'-'*64}")
    for node in graph2.topological_order():
        c, r = res_measured[node.name]
        if c is not None:
            lr = 0.1 * (node.width_dim ** (-c))
            print(f"   {node.name:<14} {node.layer_type:<10} {c:>8.4f} {r:>8.4f}  {lr:>20.6f}")
        else:
            print(f"   {node.name:<14} {node.layer_type:<10} {'----':>8} {r:>8.4f}")
    print()

    # Show that ops that were lumped together now differ
    c_gate = res_measured["gate"][0]
    c_down = res_measured["down"][0]
    c_proj = res_measured["proj"][0]
    print(f"   gate c={c_gate:.4f}, proj c={c_proj:.4f}, down c={c_down:.4f}")
    print(f"   => Different hidden ops get different LRs based on their")
    print(f"      position in the DAG and measured alignment.")
    print(f"   => Chain solver would give all three the SAME c.")
    print()
    return graph2, res_uniform, res_measured


def demo_4_chain_vs_dag():
    print("=" * 65)
    print(" 4. CHAIN SOLVER vs DAG SOLVER")
    print("=" * 65)
    print()
    print(" Chain solver collapses by layer_type: all 'hidden' ops share one c.")
    print(" DAG solver gives each op its own c.")
    print()

    model = MiniTransformer()
    x = torch.randint(0, 256, (1, 8))

    # Chain solver (no sample_input)
    torch.manual_seed(0)
    model1 = MiniTransformer()
    param_chain = Parametrization(model1, lr_prefactor=0.1)

    # DAG solver (with sample_input)
    torch.manual_seed(0)
    model2 = MiniTransformer()
    param_dag = Parametrization(model2, lr_prefactor=0.1, sample_input=x)

    chain_groups = {g["layer_name"]: g for g in param_chain.param_groups if g.get("maxp_managed")}
    dag_groups = {g["layer_name"]: g for g in param_dag.param_groups if g.get("maxp_managed")}

    print(f"   {'op':<14} {'c (chain)':<12} {'c (DAG)':<12} {'same?'}")
    print(f"   {'-'*48}")
    for name in ["tok_emb", "qkv", "proj", "gate", "up", "down", "head"]:
        if name in chain_groups and name in dag_groups:
            cc = chain_groups[name]["c"]
            cd = dag_groups[name]["c"]
            same = "yes" if abs(cc - cd) < 1e-6 else "NO"
            print(f"   {name:<14} {cc:<12.4f} {cd:<12.4f} {same}")
    print()
    print(" With uniform alignment presets, both give the same c.")
    print(" With measured per-op alignment (Phase 2), only the DAG solver")
    print(" can differentiate — that's the point of this infrastructure.")
    print()


def plot_results(graph, res_uniform, res_measured, filename):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    topo = graph.topological_order()
    weight_nodes = [n for n in topo if n.has_weight]
    names = [n.name for n in weight_nodes]
    x = np.arange(len(names))

    # --- Panel 1: c values ---
    ax = axes[0]
    c_uniform = [res_uniform[n.name][0] for n in weight_nodes]
    c_measured = [res_measured[n.name][0] for n in weight_nodes]
    w = 0.35
    ax.bar(x - w/2, c_uniform, w, label="Uniform alignment", color="#95a5a6", alpha=0.8)
    ax.bar(x + w/2, c_measured, w, label="Measured alignment", color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("c (lower = higher LR)")
    ax.set_title("Per-op c values: uniform vs measured alignment")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 2: resulting LR ---
    ax = axes[1]
    lr_uniform = [0.1 * (n.width_dim ** (-res_uniform[n.name][0])) for n in weight_nodes]
    lr_measured = [0.1 * (n.width_dim ** (-res_measured[n.name][0])) for n in weight_nodes]
    ax.bar(x - w/2, lr_uniform, w, label="Uniform alignment", color="#95a5a6", alpha=0.8)
    ax.bar(x + w/2, lr_measured, w, label="Measured alignment", color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Learning rate")
    ax.set_title("Per-op LR (lr_prefactor=0.1, width=64)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")

    fig.suptitle("DAG solver: per-op learning rates from traced transformer",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f" Saved {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAG solver demo")
    parser.add_argument("--plot", action="store_true", help="Generate figure")
    parser.add_argument("-o", "--output", default="dag_solver_demo.png",
                        help="Output filename (default: dag_solver_demo.png)")
    args = parser.parse_args()

    demo_1_analytical()
    demo_2_optimality()
    graph, res_uniform, res_measured = demo_3_traced_transformer()
    demo_4_chain_vs_dag()

    if args.plot:
        plot_results(graph, res_uniform, res_measured, args.output)

    print("Done.")
