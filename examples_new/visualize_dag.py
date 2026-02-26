#!/usr/bin/env python
"""Visualize the traced PM-to-PM DAG for a model.

Usage:
    python examples_new/visualize_dag.py
    python examples_new/visualize_dag.py --solve
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import torch

from maxp_new.dag import trace_pm_dag, MergeType
from maxp_new.solver import find_c_dag_adam
from examples_new.parameterize_example.parameterized_transformer import (
    Transformer as PTransformer,
)


def print_dag(graph, solution=None):
    topo = graph.topological_order()

    # ── Header ──
    print()
    print("  DAG: ParametrizedModule data flow")
    print("  " + "=" * 62)
    print()

    # ── Node list ──
    W = "W"  # has weight
    A = " "  # activation-only

    max_name = max(len(n.name) for n in topo)
    col = max(max_name + 2, 20)

    if solution:
        print(f"  {'':3} {'name':<{col}} {'type':<10} {'w':<4} {'width':>6} {'c':>8} {'r':>8}")
        print(f"  {'':3} {'-'*col} {'-'*10} {'-'*4} {'-'*6} {'-'*8} {'-'*8}")
    else:
        print(f"  {'':3} {'name':<{col}} {'type':<10} {'w':<4} {'width':>6}")
        print(f"  {'':3} {'-'*col} {'-'*10} {'-'*4} {'-'*6}")

    for i, node in enumerate(topo):
        w = W if node.has_weight else A
        if solution:
            c, r = solution[node.name]
            c_str = f"{c:.4f}" if c is not None else "  --"
            print(f"  {i:>2}. {node.name:<{col}} {node.layer_type:<10} {w:<4} {node.width_dim:>6} {c_str:>8} {r:>8.4f}")
        else:
            print(f"  {i:>2}. {node.name:<{col}} {node.layer_type:<10} {w:<4} {node.width_dim:>6}")

    # ── Edge list ──
    print()
    print("  Edges")
    print("  " + "-" * 62)

    for node in topo:
        if not node.predecessors:
            print(f"  (input) ──> {node.name}")
        else:
            preds = sorted(node.predecessors)
            if len(preds) == 1:
                print(f"  {preds[0]} ──> {node.name}")
            else:
                merge = "+" if node.merge_type == MergeType.MIN else "*"
                tag = "add=MIN" if node.merge_type == MergeType.MIN else "mul=SUM"
                parts = f" {merge} ".join(preds)
                print(f"  ({parts}) ──> {node.name}  [{tag}]")

    # ── ASCII DAG ──
    print()
    print("  Topology")
    print("  " + "-" * 62)
    print()

    # Assign each node a depth (longest path from any source)
    depth: dict[str, int] = {}
    for node in topo:
        if not node.predecessors:
            depth[node.name] = 0
        else:
            depth[node.name] = max(depth[p] for p in node.predecessors) + 1

    max_depth = max(depth.values()) if depth else 0

    for d in range(max_depth + 1):
        layer_nodes = [n for n in topo if depth[n.name] == d]
        names = []
        for n in layer_nodes:
            w = "W" if n.has_weight else " "
            merge = ""
            if len(n.predecessors) > 1:
                merge = " +" if n.merge_type == MergeType.MIN else " *"
            names.append(f"[{w}]{n.name}{merge}")
        print(f"  {d:>2}│ {'   '.join(names)}")
        if d < max_depth:
            # Show connections
            next_nodes = [n for n in topo if depth[n.name] == d + 1]
            arrows = []
            for nn in next_nodes:
                from_this_layer = [p for p in nn.predecessors if depth[p] == d]
                if from_this_layer:
                    arrows.append(f"{'|'.join(from_this_layer)} -> {nn.name}")
            if arrows:
                print(f"    │   └─ {', '.join(arrows)}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize traced DAG")
    parser.add_argument("--solve", action="store_true", help="Also solve for c values")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    args = parser.parse_args()

    model = PTransformer(
        vocab_size=256, d_model=args.d_model, n_heads=args.n_heads,
        d_ff=args.d_ff, n_layers=args.n_layers,
    )
    x = torch.randint(0, 256, (1, 8))
    graph = trace_pm_dag(model, x)

    solution = None
    if args.solve:
        solution = find_c_dag_adam(graph)

    print_dag(graph, solution)
