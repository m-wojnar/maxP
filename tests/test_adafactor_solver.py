"""Tests for find_c_adafactor.

Cross-validates the graph solver against (1) known preset values from
Everett et al. Table 1, (2) an independently derived (B, C, D) closed-form
backward recursion for the clamped Adafactor chain program, including
clamp-active instances where naive clipping is provably wrong.
"""

import random

import pytest

from maxp.dag import DagNode, OpGraph
from maxp.solver import find_c, find_c_adafactor

NEG = -1e18


def make_chain(n_hidden, aligns, ab_out=(0.5, 0.5)):
    names = ["emb"] + [f"h{i}" for i in range(n_hidden)] + ["out"]
    ab = [(-0.5, 0.5)] + [(0.0, 0.5)] * n_hidden + [ab_out]
    nodes = {}
    for i, name in enumerate(names):
        a, b = ab[i]
        al, om, u = aligns[i]
        lt = "embedding" if i == 0 else ("readout" if i == len(names) - 1 else "hidden")
        nodes[name] = DagNode(
            name=name, a=a, b=b, layer_type=lt, has_weight=True, width_dim=1024,
            predecessors=[names[i - 1]] if i > 0 else [],
            successors=[names[i + 1]] if i < len(names) - 1 else [],
            align_z0_dW=al, align_dZ_w0=om, align_dZ_dW=u,
        )
    return OpGraph(nodes), names, ab


def bcd_value(ab, aligns):
    """Closed-form optimal value (doc §6c): three-scalar backward recursion."""
    L = len(ab)
    B, C, D = 0.0, NEG, 0.0
    for l in range(L - 1, 0, -1):
        A = ab[l][0] + ab[l][1]
        al, om, u = aligns[l]
        Dt = max(D, 0.0)
        B, C, D = (max(B, B + Dt + al - A, C + al - A),
                   max(B + Dt + u - A, C + u - A, C + om - A),
                   om - A + Dt)
    return max(max(0.0, D) + B, C)


class TestAdafactorPresets:
    def test_full_alignment_mup_chain(self):
        """Full alignment (1, 1/2, 1) on muP (a,b): matches Table 1 exponents."""
        aligns = [(1.0, 0.5, 1.0)] * 4
        graph, names, ab = make_chain(2, aligns)
        res = find_c_adafactor(graph)
        # emb: c = 0 (LR 1); hidden: c = 1/2 (LR 1/sqrt(n)); readout: c = 0 (LR 1)
        assert res["emb"][0] == pytest.approx(0.0, abs=1e-6)
        assert res["h0"][0] == pytest.approx(0.5, abs=1e-6)
        assert res["h1"][0] == pytest.approx(0.5, abs=1e-6)
        assert res["out"][0] == pytest.approx(0.0, abs=1e-6)

    def test_no_alignment_mup_chain(self):
        """No alignment (1/2, 1/2, 1/2): all c = 0 (O(1) LRs everywhere)."""
        aligns = [(0.5, 0.5, 0.5)] * 4
        graph, names, ab = make_chain(2, aligns)
        res = find_c_adafactor(graph)
        for name in names:
            assert res[name][0] == pytest.approx(0.0, abs=1e-6)

    def test_dispatch(self):
        aligns = [(0.5, 0.5, 0.5)] * 3
        graph, _, _ = make_chain(1, aligns)
        res = find_c(graph, optimizer_type="adafactor")
        assert all(v[0] is not None for v in res.values())


class TestAdafactorClampCorrectness:
    def test_clamp_nonnegative(self):
        """c >= 0 enforced even when alignments would push c negative."""
        aligns = [(0.5, 0.5, 0.5), (0.2, 0.3, 0.2), (0.3, 0.3, 0.3)]
        graph, names, _ = make_chain(1, aligns)
        res = find_c_adafactor(graph)
        for name in names:
            assert res[name][0] >= -1e-9

    def test_clamp_active_not_naive_clipping(self):
        """The clamp-active counterexample: a clamped layer's forced residual
        subsidizes a downstream u-excess; naive clipping gives 0.1, LP gives 0."""
        aligns = [(0.5, 0.5, 0.5), (0.3, 0.4, 0.3), (0.5, 0.6, 1.1)]
        graph, names, ab = make_chain(1, aligns)
        res = find_c_adafactor(graph)
        total = sum(res[n][0] for n in names)
        assert total == pytest.approx(0.0, abs=1e-6)
        assert total == pytest.approx(bcd_value(ab, aligns), abs=1e-6)


class TestAdafactorClosedFormAgreement:
    def test_random_chains_match_bcd(self):
        """Graph solver == closed form on random chains (clamps pushed hard)."""
        rng = random.Random(3)
        for _ in range(60):
            nh = rng.randint(1, 5)
            L = nh + 2
            aligns = [(round(rng.uniform(0.1, 1.2), 3),
                       round(rng.uniform(0.1, 1.2), 3),
                       round(rng.uniform(0.1, 1.4), 3)) for _ in range(L)]
            graph, names, ab = make_chain(nh, aligns)
            res = find_c_adafactor(graph)
            total = sum(res[n][0] for n in names)
            assert total == pytest.approx(bcd_value(ab, aligns), abs=1e-6), aligns

    def test_readout_ab_variants(self):
        """Sink a+b > 1/2 variants also match the closed form."""
        rng = random.Random(4)
        for ab_out in [(0.5, 0.5), (1.0, 0.5), (0.25, 0.25), (0.5, 0.0)]:
            if ab_out[0] + ab_out[1] < 0.5:
                continue
            aligns = [(round(rng.uniform(0.2, 1.1), 3),
                       round(rng.uniform(0.2, 1.1), 3),
                       round(rng.uniform(0.2, 1.2), 3)) for _ in range(4)]
            graph, names, ab = make_chain(2, aligns, ab_out=ab_out)
            res = find_c_adafactor(graph)
            total = sum(res[n][0] for n in names)
            assert total == pytest.approx(bcd_value(ab, aligns), abs=1e-6)


class TestAdafactorDag:
    def test_diamond_runs_and_is_sane(self):
        """DAG smoke test: solver runs on a MIN-merge diamond; c >= 0 and every
        weighted node satisfies its x1 bound (c >= alpha - a - b + r with r >= 0
        implies c >= alpha - a - b when binding)."""
        nodes = {}
        def N(name, a, b, lt, hw, preds, succs, al=(0.5, 0.5, 0.5)):
            nodes[name] = DagNode(name=name, a=a, b=b, layer_type=lt,
                                  has_weight=hw, width_dim=1024,
                                  predecessors=list(preds), successors=list(succs),
                                  align_z0_dW=al[0], align_dZ_w0=al[1],
                                  align_dZ_dW=al[2])
        N("emb", -0.5, 0.5, "embedding", True, [], ["ha", "hb"])
        N("ha", 0.0, 0.5, "hidden", True, ["emb"], ["m"], al=(0.7, 0.5, 0.6))
        N("hb", 0.0, 0.5, "hidden", True, ["emb"], ["m"], al=(0.6, 0.5, 0.5))
        N("m", 0.0, 0.0, "merge", False, ["ha", "hb"], ["out"])
        N("out", 0.5, 0.5, "readout", True, ["m"], [], al=(0.48, 0.5, 0.73))
        graph = OpGraph(nodes)
        res = find_c_adafactor(graph)
        for name, (c, r) in res.items():
            if c is not None:
                assert c >= -1e-9
            assert r >= -1e-9
        # regime-(i)-style expectation for the hidden nodes (u <= alpha, om <= a+b):
        # c = max(alpha - a - b, 0)
        assert res["ha"][0] == pytest.approx(0.2, abs=1e-6)
        assert res["hb"][0] == pytest.approx(0.1, abs=1e-6)
        # readout u-excess with r_in = 0: c = max(alpha, u) - a - b = 0 clamped...
        # u - a - b = 0.73 - 1.0 < 0 -> clamp at 0
        assert res["out"][0] == pytest.approx(0.0, abs=1e-6)
