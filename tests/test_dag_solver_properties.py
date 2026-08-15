"""
Tests proving the DAG solver finds correct, optimal, per-op learning rates.

Three categories:
  1. Analytical correctness — solver matches known muP values on a chain
  2. Optimality — binding constraints + perturbation checks
  3. Per-op differentiation — real topologies get per-op c when alignment
     differs, and merge types are detected correctly from the computation

Run with:
    python -m pytest tests/test_dag_solver_properties.py -v
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from maxp.dag import DagNode, OpGraph, MergeType, trace_pm_dag
from maxp.module import ParametrizedModule
from tests.modules.chain_solver import find_c_adam as find_c_adam_chain
from maxp.solver import find_c_adam, find_c_sgd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chain_graph(align_z0_dW=1.0, align_dZ_w0=0.5, align_dZ_dW=1.0):
    """emb -> h -> head, standard muP a/b values."""
    return OpGraph({
        "emb":  DagNode(name="emb",  a=-0.5, b=0.5, layer_type="embedding",
                        has_weight=True, width_dim=32,
                        predecessors=[], successors=["h"],
                        align_z0_dW=align_z0_dW, align_dZ_w0=align_dZ_w0, align_dZ_dW=align_dZ_dW),
        "h":    DagNode(name="h",    a=0.0, b=0.5, layer_type="hidden",
                        has_weight=True, width_dim=32,
                        predecessors=["emb"], successors=["head"],
                        align_z0_dW=align_z0_dW, align_dZ_w0=align_dZ_w0, align_dZ_dW=align_dZ_dW),
        "head": DagNode(name="head", a=0.5, b=0.5, layer_type="readout",
                        has_weight=True, width_dim=32,
                        predecessors=["h"], successors=[],
                        align_z0_dW=align_z0_dW, align_dZ_w0=align_dZ_w0, align_dZ_dW=align_dZ_dW),
    })


def _swiglu_graph(align_z0_dW=1.0, align_dZ_w0=0.5, align_dZ_dW=1.0):
    """emb -> gate,up -> down -> head.  SUM merge on down (multiply)."""
    kw = dict(align_z0_dW=align_z0_dW, align_dZ_w0=align_dZ_w0, align_dZ_dW=align_dZ_dW)
    return OpGraph({
        "emb":  DagNode(name="emb",  a=-0.5, b=0.5, layer_type="embedding",
                        has_weight=True, width_dim=32,
                        predecessors=[], successors=["gate", "up"], **kw),
        "gate": DagNode(name="gate", a=0.0, b=0.5, layer_type="hidden",
                        has_weight=True, width_dim=32,
                        predecessors=["emb"], successors=["down"], **kw),
        "up":   DagNode(name="up",   a=0.0, b=0.5, layer_type="hidden",
                        has_weight=True, width_dim=32,
                        predecessors=["emb"], successors=["down"], **kw),
        "down": DagNode(name="down", a=0.0, b=0.5, layer_type="hidden",
                        has_weight=True, width_dim=32,
                        predecessors=["gate", "up"], successors=["head"],
                        merge_type=MergeType.SUM, **kw),
        "head": DagNode(name="head", a=0.5, b=0.5, layer_type="readout",
                        has_weight=True, width_dim=32,
                        predecessors=["down"], successors=[], **kw),
    })


def _compute_hidden_r(c, r_in, align_z0_dW=1.0, align_dZ_w0=0.5, align_dZ_dW=1.0, a=0.0, b=0.5):
    """Manually compute r for a hidden node: r = min(x1, x2, x3)."""
    x1 = a + c - align_z0_dW
    x2 = a + c + r_in - align_dZ_dW
    x3 = (a + b) + r_in - align_dZ_w0
    return min(x1, x2, x3)


def _compute_readout_r(c, r_in, align_z0_dW=1.0, align_dZ_w0=0.5, align_dZ_dW=1.0, a=0.5, b=0.5):
    """Manually compute r for a readout node."""
    x1 = (a + b) + r_in - align_dZ_w0
    x2 = a + c - align_z0_dW
    x3 = a + c + r_in - align_dZ_dW
    return min(x1, x2, x3)


# ═══════════════════════════════════════════════════════════════════════════
# 1. ANALYTICAL CORRECTNESS
# ═══════════════════════════════════════════════════════════════════════════

class TestAnalyticalCorrectness:

    def test_mup_chain_exact_values(self):
        """Standard 3-node muP chain: known solution c = [0.5, 1.0, 0.5]."""
        res = find_c_adam(_chain_graph())
        assert abs(res["emb"][0] - 0.5) < 1e-6
        assert abs(res["h"][0] - 1.0) < 1e-6
        assert abs(res["head"][0] - 0.5) < 1e-6

    def test_dag_matches_flat_solver_full_alignment(self):
        """Graph solver on a linear chain matches the flat chain solver exactly."""
        res = find_c_adam(_chain_graph())
        cl_flat, rl_flat = find_c_adam_chain(
            [-0.5, 0.0, 0.5], [0.5, 0.5, 0.5],
            [1.0]*3, [0.5]*3, [1.0]*3,
        )
        for (name, idx) in [("emb", 0), ("h", 1), ("head", 2)]:
            assert abs(res[name][0] - cl_flat[idx]) < 1e-6
            assert abs(res[name][1] - rl_flat[idx]) < 1e-6

    def test_dag_matches_flat_solver_no_alignment(self):
        """Same check with no-alignment preset."""
        res = find_c_adam(_chain_graph(align_z0_dW=0, align_dZ_w0=0, align_dZ_dW=0))
        cl_flat, rl_flat = find_c_adam_chain(
            [-0.5, 0.0, 0.5], [0.5, 0.5, 0.5],
            [0.0]*3, [0.0]*3, [0.0]*3,
        )
        for (name, idx) in [("emb", 0), ("h", 1), ("head", 2)]:
            assert abs(res[name][0] - cl_flat[idx]) < 1e-6

    def test_emb_c_equals_neg_a(self):
        """Source node: r = a + c >= 0 => c >= -a. Solver should hit equality."""
        res = find_c_adam(_chain_graph())
        a_emb = -0.5
        assert abs(res["emb"][0] - (-a_emb)) < 1e-6  # c = 0.5 = -(-0.5)


# ═══════════════════════════════════════════════════════════════════════════
# 2. OPTIMALITY
# ═══════════════════════════════════════════════════════════════════════════

class TestOptimality:

    def test_all_r_nonneg(self):
        """Basic feasibility: every r >= 0."""
        res = find_c_adam(_chain_graph())
        for name, (c, r) in res.items():
            assert r >= -1e-8, f"r_{name} = {r} < 0"

    def test_all_binding_full_alignment(self):
        """With full alignment, all r = 0 (every constraint binding)."""
        res = find_c_adam(_chain_graph())
        for name, (c, r) in res.items():
            assert abs(r) < 1e-6, f"r_{name} = {r}, expected 0 (binding)"

    def test_perturb_c_emb_violates(self):
        """Reducing c_emb by eps makes r_emb < 0."""
        res = find_c_adam(_chain_graph())
        c_emb_opt = res["emb"][0]
        r_emb_perturbed = -0.5 + (c_emb_opt - 0.01)
        assert r_emb_perturbed < 0

    def test_perturb_c_hidden_violates(self):
        """Reducing c_h by eps makes r_h < 0."""
        res = find_c_adam(_chain_graph())
        c_h_opt = res["h"][0]
        r_h_perturbed = _compute_hidden_r(c_h_opt - 0.01, r_in=0.0)
        assert r_h_perturbed < 0

    def test_perturb_c_head_violates(self):
        """Reducing c_head by eps makes r_head < 0."""
        res = find_c_adam(_chain_graph())
        c_head_opt = res["head"][0]
        r_head_perturbed = _compute_readout_r(c_head_opt - 0.01, r_in=0.0)
        assert r_head_perturbed < 0

    def test_sum_c_is_minimal(self):
        """Each c is at its individual lower bound (no slack)."""
        res = find_c_adam(_chain_graph())
        assert abs(res["emb"][0] - 0.5) < 1e-6
        assert abs(res["h"][0] - 1.0) < 1e-6
        assert abs(res["head"][0] - 0.5) < 1e-6

    def test_swiglu_all_r_nonneg(self):
        """SwiGLU graph: all r >= 0."""
        res = find_c_adam(_swiglu_graph())
        for name, (c, r) in res.items():
            assert r >= -1e-8, f"r_{name} = {r} < 0"

    def test_swiglu_all_binding_full_alignment(self):
        """SwiGLU with full alignment: all r = 0 (optimal under preset alignment)."""
        res = find_c_adam(_swiglu_graph())
        for name, (c, r) in res.items():
            assert abs(r) < 1e-6, f"r_{name} = {r}, expected 0"

    def test_sgd_all_r_nonneg(self):
        """SGD on a chain: all r >= 0."""
        res = find_c_sgd(_chain_graph())
        for name, (c, r) in res.items():
            assert r >= -1e-8, f"r_{name} = {r} < 0"


# ═══════════════════════════════════════════════════════════════════════════
# 3. PER-OP DIFFERENTIATION
# ═══════════════════════════════════════════════════════════════════════════

class TestPerOpDifferentiation:

    def test_swiglu_per_op_c_with_measured_alignment(self):
        """With per-op alignment values, gate/up/down get different c.

        This is the core value of the DAG solver: when dynamic measurement finds
        different alignment per op, the solver assigns different c per op.
        The chain solver can't do this — it collapses all 'hidden' ops
        into one c value.

        Key insight: changing align_z0_dW alone on `down` won't differentiate c
        when align_dZ_dW >= align_z0_dW and r_in = 0, because
        x2 = c + r_in - align_dZ_dW still dominates. We also adjust
        align_dZ_dW so the align_z0_dW constraint can bind.
        """
        # Simulate measured alignment: down has weaker alignment
        g = _swiglu_graph()
        g.nodes["down"].align_z0_dW = 0.5
        g.nodes["down"].align_dZ_dW = 0.5  # weaker cross-term too

        res = find_c_adam(g)

        # gate and up should get the same c (same predecessors, same alignment)
        assert abs(res["gate"][0] - res["up"][0]) < 1e-6

        # down should get a DIFFERENT c than gate/up
        # With align_z0_dW=0.5, align_dZ_dW=0.5 on down:
        #   x1 = c - 0.5, x2 = c + r_in - 0.5
        # Both allow c = 0.5 (vs gate's c = 1.0 from align_z0_dW=1.0, align_dZ_dW=1.0)
        assert abs(res["down"][0] - res["gate"][0]) > 0.1

    def test_chain_solver_collapses_but_dag_differentiates(self):
        """Chain solver gives all hidden ops the same c.
        DAG solver can differentiate when alignment differs per op."""
        g = _swiglu_graph()
        # Give different alignment to each hidden op
        g.nodes["gate"].align_z0_dW = 0.8
        g.nodes["up"].align_z0_dW = 0.8
        g.nodes["down"].align_z0_dW = 0.3

        res = find_c_adam(g)

        c_gate = res["gate"][0]
        c_down = res["down"][0]

        # They must be different (different align_z0_dW)
        assert abs(c_gate - c_down) > 0.1

        # Chain solver would give them all the same c — that's the limitation
        # the DAG solver fixes.

    def test_residual_detected_as_min_merge(self):
        """Tracing a model with x + linear(x) detects MIN merge."""
        class ResModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = ParametrizedModule(
                    nn.Embedding(64, 32), width_dim=32, layer_type="embedding")
                self.linear = ParametrizedModule(
                    nn.Linear(32, 32, bias=False), width_dim=32, layer_type="hidden")
                self.head = ParametrizedModule(
                    nn.Linear(32, 64, bias=False), width_dim=32, layer_type="readout")

            def forward(self, x):
                h = self.emb(x)
                return self.head(h + self.linear(h))  # residual = addition = MIN

        model = ResModel()
        x = torch.randint(0, 64, (1, 4))
        graph = trace_pm_dag(model, x)

        # Addition creates a synthetic MIN merge node
        add_nodes = [n for n in graph.nodes if n.startswith("_add_")]
        assert len(add_nodes) == 1
        assert graph.nodes[add_nodes[0]].merge_type == MergeType.MIN

    def test_multiply_detected_as_sum_merge(self):
        """Tracing a model with gate(x) * up(x) detects SUM merge."""
        class SwiGLUModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = ParametrizedModule(
                    nn.Embedding(64, 32), width_dim=32, layer_type="embedding")
                self.gate = ParametrizedModule(
                    nn.Linear(32, 64, bias=False), width_dim=32, layer_type="hidden")
                self.up = ParametrizedModule(
                    nn.Linear(32, 64, bias=False), width_dim=32, layer_type="hidden")
                self.down = ParametrizedModule(
                    nn.Linear(64, 32, bias=False), width_dim=64, layer_type="hidden")
                self.head = ParametrizedModule(
                    nn.Linear(32, 64, bias=False), width_dim=32, layer_type="readout")

            def forward(self, x):
                h = self.emb(x)
                return self.head(self.down(F.silu(self.gate(h)) * self.up(h)))

        model = SwiGLUModel()
        x = torch.randint(0, 64, (1, 4))
        graph = trace_pm_dag(model, x)

        # gate * up creates a synthetic SUM merge node
        mul_nodes = [n for n in graph.nodes if n.startswith("_mul_")]
        assert len(mul_nodes) == 1
        assert graph.nodes[mul_nodes[0]].merge_type == MergeType.SUM

    def test_constraint_math_manual_verification(self):
        """Manually verify the solver's r values satisfy the constraint formulas."""
        g = _swiglu_graph()
        # Use non-trivial alignment to make constraints interesting
        g.nodes["gate"].align_z0_dW = 0.7
        g.nodes["up"].align_z0_dW = 0.7
        g.nodes["down"].align_z0_dW = 0.4

        res = find_c_adam(g)

        # Emb: r = a + c
        r_emb = res["emb"][1]
        assert abs(r_emb - (-0.5 + res["emb"][0])) < 1e-6

        # Gate: r = min(x1, x2, x3) with r_in = r_emb
        c_gate = res["gate"][0]
        r_gate_manual = _compute_hidden_r(
            c_gate, r_emb, align_z0_dW=0.7, align_dZ_w0=0.5, align_dZ_dW=1.0)
        assert abs(res["gate"][1] - r_gate_manual) < 1e-6

        # Up: same structure as gate
        c_up = res["up"][0]
        r_up_manual = _compute_hidden_r(
            c_up, r_emb, align_z0_dW=0.7, align_dZ_w0=0.5, align_dZ_dW=1.0)
        assert abs(res["up"][1] - r_up_manual) < 1e-6

        # Down: SUM merge, r_in = r_gate + r_up
        r_in_down = res["gate"][1] + res["up"][1]
        c_down = res["down"][0]
        r_down_manual = _compute_hidden_r(
            c_down, r_in_down, align_z0_dW=0.4, align_dZ_w0=0.5, align_dZ_dW=1.0)
        assert abs(res["down"][1] - r_down_manual) < 1e-6

        # Head: r_in = r_down
        r_in_head = res["down"][1]
        c_head = res["head"][0]
        r_head_manual = _compute_readout_r(c_head, r_in_head)
        assert abs(res["head"][1] - r_head_manual) < 1e-6

    def test_feature_learning_tightens_constraints(self):
        """feature_learning=True forces r=0 at pre-sink nodes, increasing sum(c)."""
        g = _chain_graph(align_z0_dW=0, align_dZ_w0=0, align_dZ_dW=0)  # no-alignment for slack
        res_no_fl = find_c_adam(g, feature_learning=False)
        res_fl = find_c_adam(g, feature_learning=True)

        sum_no_fl = sum(c for c, _ in res_no_fl.values() if c is not None)
        sum_fl = sum(c for c, _ in res_fl.values() if c is not None)

        # feature_learning should be at least as tight
        assert sum_fl >= sum_no_fl - 1e-8
        # Pre-sink r should be 0
        assert abs(res_fl["h"][1]) < 1e-6
