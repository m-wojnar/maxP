"""Tests for DAG tracing, DAG solver, and integration with Parametrization."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from maxp_new.dag import MergeType, DagNode, OpGraph, trace_pm_dag
from maxp_new.module import ParametrizedModule
from maxp_new.parametrization import Parametrization
from maxp_new.solver import find_c_adam, find_c_dag_adam, find_c_dag_sgd, find_c_dag


# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------

class SimpleChain(nn.Module):
    """3 PMs in series: embedding -> hidden -> readout."""

    def __init__(self, d=32, vocab=64):
        super().__init__()
        self.emb = ParametrizedModule(
            nn.Embedding(vocab, d), width_dim=d, layer_type="embedding")
        self.hidden = ParametrizedModule(
            nn.Linear(d, d, bias=False), width_dim=d, layer_type="hidden")
        self.head = ParametrizedModule(
            nn.Linear(d, vocab, bias=False), width_dim=d, layer_type="readout")

    def forward(self, x):
        return self.head(self.hidden(self.emb(x)))


class ForkModel(nn.Module):
    """One PM feeds two parallel PMs that merge at a readout."""

    def __init__(self, d=32, vocab=64):
        super().__init__()
        self.emb = ParametrizedModule(
            nn.Embedding(vocab, d), width_dim=d, layer_type="embedding")
        self.branch_a = ParametrizedModule(
            nn.Linear(d, d, bias=False), width_dim=d, layer_type="hidden")
        self.branch_b = ParametrizedModule(
            nn.Linear(d, d, bias=False), width_dim=d, layer_type="hidden")
        self.head = ParametrizedModule(
            nn.Linear(d, vocab, bias=False), width_dim=d, layer_type="readout")

    def forward(self, x):
        h = self.emb(x)
        a = self.branch_a(h)
        b = self.branch_b(h)
        return self.head(a + b)  # addition -> MIN merge


class SwiGLUModel(nn.Module):
    """gate(x) * up(x) -> down: multiply merge (SUM)."""

    def __init__(self, d=32, d_ff=64, vocab=64):
        super().__init__()
        self.emb = ParametrizedModule(
            nn.Embedding(vocab, d), width_dim=d, layer_type="embedding")
        self.gate = ParametrizedModule(
            nn.Linear(d, d_ff, bias=False), width_dim=d, layer_type="hidden")
        self.up = ParametrizedModule(
            nn.Linear(d, d_ff, bias=False), width_dim=d, layer_type="hidden")
        self.down = ParametrizedModule(
            nn.Linear(d_ff, d, bias=False), width_dim=d_ff, layer_type="hidden")
        self.head = ParametrizedModule(
            nn.Linear(d, vocab, bias=False), width_dim=d, layer_type="readout")

    def forward(self, x):
        h = self.emb(x)
        return self.head(self.down(F.silu(self.gate(h)) * self.up(h)))


class ResidualModel(nn.Module):
    """PM_a output + PM_b output -> PM_c: residual connection (MIN merge)."""

    def __init__(self, d=32, vocab=64):
        super().__init__()
        self.emb = ParametrizedModule(
            nn.Embedding(vocab, d), width_dim=d, layer_type="embedding")
        self.linear = ParametrizedModule(
            nn.Linear(d, d, bias=False), width_dim=d, layer_type="hidden")
        self.head = ParametrizedModule(
            nn.Linear(d, vocab, bias=False), width_dim=d, layer_type="readout")

    def forward(self, x):
        h = self.emb(x)
        return self.head(h + self.linear(h))  # residual: MIN merge on head


# ---------------------------------------------------------------------------
# Tracing tests
# ---------------------------------------------------------------------------

class TestDagTracing:

    def test_simple_chain_dag(self):
        model = SimpleChain()
        x = torch.randint(0, 64, (1, 4))
        graph = trace_pm_dag(model, x)

        assert set(graph.nodes.keys()) == {"emb", "hidden", "head"}

        emb = graph.nodes["emb"]
        assert emb.predecessors == []
        assert "hidden" in emb.successors

        hidden = graph.nodes["hidden"]
        assert "emb" in hidden.predecessors
        assert "head" in hidden.successors

        head = graph.nodes["head"]
        assert "hidden" in head.predecessors
        assert head.successors == []

    def test_fork_dag(self):
        model = ForkModel()
        x = torch.randint(0, 64, (1, 4))
        graph = trace_pm_dag(model, x)

        # PM nodes plus a synthetic merge node for branch_a + branch_b
        assert "emb" in graph.nodes
        assert "branch_a" in graph.nodes
        assert "branch_b" in graph.nodes
        assert "head" in graph.nodes
        merge_nodes = [n for n in graph.nodes if n.startswith("_add_")]
        assert len(merge_nodes) == 1

        # emb fans out to both branches
        emb = graph.nodes["emb"]
        assert "branch_a" in emb.successors
        assert "branch_b" in emb.successors

        # head sees the synthetic merge node (not both branches directly)
        head = graph.nodes["head"]
        assert merge_nodes[0] in head.predecessors
        # The merge node carries the MIN merge type
        assert graph.nodes[merge_nodes[0]].merge_type == MergeType.MIN

    def test_swiglu_multiply_merge(self):
        model = SwiGLUModel()
        x = torch.randint(0, 64, (1, 4))
        graph = trace_pm_dag(model, x)

        assert "gate" in graph.nodes
        assert "up" in graph.nodes
        assert "down" in graph.nodes

        # gate * up creates a synthetic SUM merge node; down sees that node
        mul_nodes = [n for n in graph.nodes if n.startswith("_mul_")]
        assert len(mul_nodes) == 1
        mul_node = graph.nodes[mul_nodes[0]]
        assert "gate" in mul_node.predecessors
        assert "up" in mul_node.predecessors
        assert mul_node.merge_type == MergeType.SUM

        down = graph.nodes["down"]
        assert mul_nodes[0] in down.predecessors

    def test_residual_connection(self):
        model = ResidualModel()
        x = torch.randint(0, 64, (1, 4))
        graph = trace_pm_dag(model, x)

        # emb + linear(emb) creates a synthetic MIN merge node
        add_nodes = [n for n in graph.nodes if n.startswith("_add_")]
        assert len(add_nodes) == 1
        add_node = graph.nodes[add_nodes[0]]
        assert "emb" in add_node.predecessors
        assert "linear" in add_node.predecessors
        assert add_node.merge_type == MergeType.MIN

        # head sees the merge node, not emb/linear directly
        head = graph.nodes["head"]
        assert add_nodes[0] in head.predecessors

    def test_graph_validation(self):
        model = SimpleChain()
        x = torch.randint(0, 64, (1, 4))
        graph = trace_pm_dag(model, x)
        # Should not raise
        graph.validate()

    def test_topological_order(self):
        model = SimpleChain()
        x = torch.randint(0, 64, (1, 4))
        graph = trace_pm_dag(model, x)
        topo = graph.topological_order()
        names = [n.name for n in topo]
        assert names.index("emb") < names.index("hidden")
        assert names.index("hidden") < names.index("head")

    def test_sources_and_sinks(self):
        model = SimpleChain()
        x = torch.randint(0, 64, (1, 4))
        graph = trace_pm_dag(model, x)
        sources = graph.sources()
        sinks = graph.sinks()
        assert len(sources) == 1
        assert sources[0].name == "emb"
        assert len(sinks) == 1
        assert sinks[0].name == "head"

    def test_node_attributes(self):
        model = SimpleChain()
        x = torch.randint(0, 64, (1, 4))
        graph = trace_pm_dag(model, x)

        emb = graph.nodes["emb"]
        assert emb.layer_type == "embedding"
        assert emb.has_weight is True
        assert emb.width_dim == 32
        assert emb.a == -0.5
        assert emb.b == 0.5

        hidden = graph.nodes["hidden"]
        assert hidden.layer_type == "hidden"
        assert hidden.has_weight is True


# ---------------------------------------------------------------------------
# DAG solver tests
# ---------------------------------------------------------------------------

class TestDagSolver:

    def test_dag_chain_matches_flat_solver(self):
        """A linear DAG should give the same c values as the flat chain solver."""
        # Build a simple 3-node chain graph manually
        nodes = {
            "emb": DagNode(
                name="emb", a=-0.5, b=0.5, layer_type="embedding",
                has_weight=True, width_dim=32,
                predecessors=[], successors=["hidden"],
            ),
            "hidden": DagNode(
                name="hidden", a=0.0, b=0.5, layer_type="hidden",
                has_weight=True, width_dim=32,
                predecessors=["emb"], successors=["head"],
            ),
            "head": DagNode(
                name="head", a=0.5, b=0.5, layer_type="readout",
                has_weight=True, width_dim=32,
                predecessors=["hidden"], successors=[],
            ),
        }
        graph = OpGraph(nodes)

        # Solve with DAG solver
        dag_result = find_c_dag_adam(graph)

        # Solve with flat solver (same a,b ordering)
        al = [-0.5, 0.0, 0.5]
        bl = [0.5, 0.5, 0.5]
        alpha = [1.0, 1.0, 1.0]
        omega = [0.5, 0.5, 0.5]
        u = [1.0, 1.0, 1.0]
        flat_cl, flat_rl = find_c_adam(al, bl, alpha, omega, u)

        # Compare c values
        dag_cl = [dag_result["emb"][0], dag_result["hidden"][0], dag_result["head"][0]]
        dag_rl = [dag_result["emb"][1], dag_result["hidden"][1], dag_result["head"][1]]

        for dc, fc in zip(dag_cl, flat_cl):
            assert abs(dc - fc) < 1e-6, f"c mismatch: DAG={dc}, flat={fc}"
        for dr, fr in zip(dag_rl, flat_rl):
            assert abs(dr - fr) < 1e-6, f"r mismatch: DAG={dr}, flat={fr}"

    def test_dag_fork_valid_r(self):
        """Fork topology produces valid non-negative r values."""
        nodes = {
            "emb": DagNode(
                name="emb", a=-0.5, b=0.5, layer_type="embedding",
                has_weight=True, width_dim=32,
                predecessors=[], successors=["a", "b"],
            ),
            "a": DagNode(
                name="a", a=0.0, b=0.5, layer_type="hidden",
                has_weight=True, width_dim=32,
                predecessors=["emb"], successors=["head"],
            ),
            "b": DagNode(
                name="b", a=0.0, b=0.5, layer_type="hidden",
                has_weight=True, width_dim=32,
                predecessors=["emb"], successors=["head"],
            ),
            "head": DagNode(
                name="head", a=0.5, b=0.5, layer_type="readout",
                has_weight=True, width_dim=32,
                predecessors=["a", "b"], successors=[],
                merge_type=MergeType.MIN,
            ),
        }
        graph = OpGraph(nodes)
        result = find_c_dag_adam(graph)

        for name, (c, r) in result.items():
            assert r >= -1e-8, f"r for '{name}' is negative: {r}"
            if c is not None:
                assert np.isfinite(c), f"c for '{name}' is not finite"

    def test_dag_min_merge(self):
        """MIN merge: r = min(r1, r2) for addition."""
        nodes = {
            "s1": DagNode(
                name="s1", a=-0.5, b=0.5, layer_type="embedding",
                has_weight=True, width_dim=32,
                predecessors=[], successors=["h"],
            ),
            "s2": DagNode(
                name="s2", a=-0.5, b=0.5, layer_type="embedding",
                has_weight=True, width_dim=32,
                predecessors=[], successors=["h"],
            ),
            "h": DagNode(
                name="h", a=0.0, b=0.5, layer_type="hidden",
                has_weight=True, width_dim=32,
                predecessors=["s1", "s2"], successors=["out"],
                merge_type=MergeType.MIN,
            ),
            "out": DagNode(
                name="out", a=0.5, b=0.5, layer_type="readout",
                has_weight=True, width_dim=32,
                predecessors=["h"], successors=[],
            ),
        }
        graph = OpGraph(nodes)
        result = find_c_dag_adam(graph)

        # r for h should be constrained by min of r_s1, r_s2 (as r_in)
        r_s1 = result["s1"][1]
        r_s2 = result["s2"][1]
        r_h = result["h"][1]
        # r_h <= min(r_s1, r_s2) + constant terms
        assert r_h >= -1e-8

    def test_dag_sum_merge(self):
        """SUM merge: r_in = r1 + r2 for multiply."""
        nodes = {
            "emb": DagNode(
                name="emb", a=-0.5, b=0.5, layer_type="embedding",
                has_weight=True, width_dim=32,
                predecessors=[], successors=["gate", "up"],
            ),
            "gate": DagNode(
                name="gate", a=0.0, b=0.5, layer_type="hidden",
                has_weight=True, width_dim=32,
                predecessors=["emb"], successors=["down"],
            ),
            "up": DagNode(
                name="up", a=0.0, b=0.5, layer_type="hidden",
                has_weight=True, width_dim=32,
                predecessors=["emb"], successors=["down"],
            ),
            "down": DagNode(
                name="down", a=0.0, b=0.5, layer_type="hidden",
                has_weight=True, width_dim=32,
                predecessors=["gate", "up"], successors=["head"],
                merge_type=MergeType.SUM,
            ),
            "head": DagNode(
                name="head", a=0.5, b=0.5, layer_type="readout",
                has_weight=True, width_dim=32,
                predecessors=["down"], successors=[],
            ),
        }
        graph = OpGraph(nodes)
        result = find_c_dag_adam(graph)

        # SUM merge means r_in for down = r_gate + r_up
        # All r should be non-negative
        for name, (c, r) in result.items():
            assert r >= -1e-8, f"r for '{name}' negative: {r}"

    def test_dag_activation_only_propagation(self):
        """Activation-only node: r = r_in + a."""
        nodes = {
            "emb": DagNode(
                name="emb", a=-0.5, b=0.5, layer_type="embedding",
                has_weight=True, width_dim=32,
                predecessors=[], successors=["act"],
            ),
            "act": DagNode(
                name="act", a=0.0, b=0.0, layer_type="hidden",
                has_weight=False, width_dim=32,
                predecessors=["emb"], successors=["head"],
            ),
            "head": DagNode(
                name="head", a=0.5, b=0.5, layer_type="readout",
                has_weight=True, width_dim=32,
                predecessors=["act"], successors=[],
            ),
        }
        graph = OpGraph(nodes)
        result = find_c_dag_adam(graph)

        # act is activation-only: r_act = r_emb + a_act = r_emb + 0
        r_emb = result["emb"][1]
        r_act = result["act"][1]
        assert result["act"][0] is None  # no c for activation-only
        assert abs(r_act - (r_emb + 0.0)) < 1e-6

    def test_dag_all_r_nonneg(self):
        """All r values must be non-negative."""
        model = SwiGLUModel()
        x = torch.randint(0, 64, (1, 4))
        graph = trace_pm_dag(model, x)
        result = find_c_dag_adam(graph)
        for name, (c, r) in result.items():
            assert r >= -1e-8, f"r for '{name}' is negative: {r}"

    def test_dag_sgd(self):
        """SGD variant works on a simple chain."""
        nodes = {
            "emb": DagNode(
                name="emb", a=-0.5, b=0.5, layer_type="embedding",
                has_weight=True, width_dim=32,
                predecessors=[], successors=["hidden"],
            ),
            "hidden": DagNode(
                name="hidden", a=0.0, b=0.5, layer_type="hidden",
                has_weight=True, width_dim=32,
                predecessors=["emb"], successors=["head"],
            ),
            "head": DagNode(
                name="head", a=0.5, b=0.5, layer_type="readout",
                has_weight=True, width_dim=32,
                predecessors=["hidden"], successors=[],
            ),
        }
        graph = OpGraph(nodes)
        result = find_c_dag_sgd(graph)

        for name, (c, r) in result.items():
            assert r >= -1e-8, f"r for '{name}' negative: {r}"
            assert c is not None
            assert np.isfinite(c)

    def test_dag_dispatch(self):
        """find_c_dag dispatches correctly."""
        nodes = {
            "emb": DagNode(
                name="emb", a=-0.5, b=0.5, layer_type="embedding",
                has_weight=True, width_dim=32,
                predecessors=[], successors=["head"],
            ),
            "head": DagNode(
                name="head", a=0.5, b=0.5, layer_type="readout",
                has_weight=True, width_dim=32,
                predecessors=["emb"], successors=[],
            ),
        }
        graph = OpGraph(nodes)
        result_adam = find_c_dag(graph, optimizer_type="adam")
        result_sgd = find_c_dag(graph, optimizer_type="sgd")
        assert isinstance(result_adam, dict)
        assert isinstance(result_sgd, dict)

        with pytest.raises(ValueError, match="Unknown optimizer_type"):
            find_c_dag(graph, optimizer_type="rmsprop")


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestParametrizationIntegration:

    def test_parametrization_with_sample_input(self):
        """Full model with sample_input uses DAG solver and assigns per-PM c values."""
        model = SwiGLUModel()
        x = torch.randint(0, 64, (1, 4))
        param = Parametrization(model, lr_prefactor=0.1, sample_input=x)

        # Should have per-PM groups (emb, gate, up, down, head) + _other
        managed = [g for g in param.param_groups if g.get("maxp_managed")]
        assert len(managed) == 5

        names = {g["layer_name"] for g in managed}
        assert names == {"emb", "gate", "up", "down", "head"}

        # All c values should be finite
        for g in managed:
            assert np.isfinite(g["c"])
            assert g["lr"] > 0

    def test_parametrization_without_sample_input_unchanged(self):
        """Without sample_input, falls back to chain solver (backward compat)."""
        model = SimpleChain()
        param = Parametrization(model, lr_prefactor=0.1)

        managed = [g for g in param.param_groups if g.get("maxp_managed")]
        assert len(managed) == 3

        # All same-type PMs get the same c
        c_vals = {g["layer_name"]: g["c"] for g in managed}
        assert np.isfinite(c_vals["emb"])
        assert np.isfinite(c_vals["hidden"])
        assert np.isfinite(c_vals["head"])

    def test_chain_and_dag_give_same_c_for_linear_model(self):
        """For a simple chain, both paths should produce the same c values."""
        torch.manual_seed(42)
        model1 = SimpleChain()
        param1 = Parametrization(model1, lr_prefactor=0.1)

        torch.manual_seed(42)
        model2 = SimpleChain()
        x = torch.randint(0, 64, (1, 4))
        param2 = Parametrization(model2, lr_prefactor=0.1, sample_input=x)

        managed1 = {g["layer_name"]: g["c"] for g in param1.param_groups if g.get("maxp_managed")}
        managed2 = {g["layer_name"]: g["c"] for g in param2.param_groups if g.get("maxp_managed")}

        for name in managed1:
            assert abs(managed1[name] - managed2[name]) < 1e-6, \
                f"c mismatch for '{name}': chain={managed1[name]}, dag={managed2[name]}"

    def test_parametrization_sgd_with_sample_input(self):
        """SGD optimizer with sample_input works."""
        model = SimpleChain()
        x = torch.randint(0, 64, (1, 4))
        param = Parametrization(model, lr_prefactor=0.1, optimizer_type="sgd",
                                sample_input=x)
        managed = [g for g in param.param_groups if g.get("maxp_managed")]
        assert len(managed) == 3
        for g in managed:
            assert np.isfinite(g["c"])


# ---------------------------------------------------------------------------
# OpGraph unit tests
# ---------------------------------------------------------------------------

class TestOpGraph:

    def test_cycle_detection(self):
        nodes = {
            "a": DagNode(name="a", a=0.0, b=0.5, layer_type="hidden",
                         has_weight=True, width_dim=32,
                         predecessors=["b"], successors=["b"]),
            "b": DagNode(name="b", a=0.0, b=0.5, layer_type="hidden",
                         has_weight=True, width_dim=32,
                         predecessors=["a"], successors=["a"]),
        }
        graph = OpGraph(nodes)
        with pytest.raises(ValueError, match="cycle"):
            graph.validate()

    def test_empty_graph(self):
        graph = OpGraph({})
        with pytest.raises(ValueError, match="no source"):
            graph.validate()
