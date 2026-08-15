"""Unit tests for the downstream-eval scoring core.

CPU-only, no torchtitan, no datasets: model and tokenizer are deterministic
fakes. The model is position-local (logits at p depend only on token at p),
which is causal, so right-padding invariance is exactly testable.
"""

import math

import pytest
import torch
import torch.nn as nn

import os
import sys

# eval_tasks lives with the experiment pipeline (not the installed package),
# so the test adds its directory to the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "lm"))

from eval_tasks import (  # noqa: E402
    ScoredItem,
    lambada_metrics,
    mc_doc_pairs,
    multiple_choice_metrics,
    perplexity_over_tokens,
    score_batch,
    score_requests,
    split_context_continuation,
)

V = 64


class SimpleTok:
    """Char-level: concatenation property holds exactly (no boundary merges)."""
    bos_id = 1
    eos_id = 2

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [3 + (ord(c) % 50) for c in text]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids


class MergeTok(SimpleTok):
    """Like SimpleTok, but the pair 'ab' merges into a single token 60 —
    simulates a BPE boundary merge between context and continuation."""

    def encode(self, text, add_bos=False, add_eos=False):
        ids, i = [], 0
        while i < len(text):
            if text[i : i + 2] == "ab":
                ids.append(60)
                i += 2
            else:
                ids.append(3 + (ord(text[i]) % 50))
                i += 1
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids


class UniformModel(nn.Module):
    """Zero logits everywhere: every token has log-prob −log(V)."""

    def forward(self, toks):
        return torch.zeros(*toks.shape, V)


class NextTokenModel(nn.Module):
    """logits at position p put all mass on (token_p + 1): the greedy
    continuation of token t is exactly t+1. Position-local, hence causal."""

    def forward(self, toks):
        out = torch.full((*toks.shape, V), -10.0)
        nxt = (toks + 1).clamp(max=V - 1)
        out.scatter_(-1, nxt.unsqueeze(-1), 10.0)
        return out


def test_split_no_merge_is_exact():
    tok = SimpleTok()
    ctx_ids, cont_ids = split_context_continuation(tok, "hello", " world")
    assert ctx_ids == tok.encode("hello")
    assert ctx_ids + cont_ids == tok.encode("hello world")


def test_split_boundary_merge_backs_off():
    tok = MergeTok()
    ctx_ids, cont_ids = split_context_continuation(tok, "za", "bc")
    full = tok.encode("zabc")            # [z, 60, c] — 'a'+'b' merged
    assert ctx_ids + cont_ids == full
    assert len(cont_ids) >= 1
    assert 60 in cont_ids                # merged token lands in the continuation


def test_uniform_logprob_and_counts():
    model, tok = UniformModel(), SimpleTok()
    [s] = score_requests(model, tok, [("abc", "de")],
                         device="cpu", batch_size=1, seq_len=16)
    assert s.n_tokens == 2
    assert s.logprob == pytest.approx(-2 * math.log(V), rel=1e-5)


def test_greedy_flag():
    model, tok = NextTokenModel(), SimpleTok()
    # Craft ids directly: context ends in t; continuation [t+1, t+2] is greedy.
    good = ([5, 6], [7, 8])
    bad = ([5, 6], [9, 8])
    res = score_batch(model, [good, bad], pad_id=0, seq_len=12, device="cpu")
    assert res[0].greedy is True
    assert res[1].greedy is False


def test_right_padding_and_batch_invariance():
    model, tok = NextTokenModel(), SimpleTok()
    pairs = [("abc", "de"), ("abcdefgh", " longer continuation x")]
    solo = [score_requests(model, tok, [p], device="cpu", batch_size=1, seq_len=64)[0]
            for p in pairs]
    together = score_requests(model, tok, pairs, device="cpu", batch_size=2, seq_len=64)
    wide = score_requests(model, tok, pairs, device="cpu", batch_size=2, seq_len=48)
    for a, b, c in zip(solo, together, wide):
        assert a.logprob == pytest.approx(b.logprob, abs=1e-6)
        assert a.logprob == pytest.approx(c.logprob, abs=1e-6)
        assert a.n_tokens == b.n_tokens == c.n_tokens


def test_partial_batch_padding_rows_dropped():
    model, tok = UniformModel(), SimpleTok()
    pairs = [("aa", "b"), ("cc", "d"), ("ee", "f")]
    res = score_requests(model, tok, pairs, device="cpu", batch_size=4, seq_len=16)
    assert len(res) == 3
    for s in res:
        assert s.logprob == pytest.approx(-math.log(V), rel=1e-5)


def test_left_truncation_keeps_continuation():
    model, tok = UniformModel(), SimpleTok()
    [s] = score_requests(model, tok, [("x" * 100, "yz")],
                         device="cpu", batch_size=1, seq_len=32)
    assert s.n_tokens == 2
    assert s.logprob == pytest.approx(-2 * math.log(V), rel=1e-5)


def test_multiple_choice_metrics_acc_and_norm():
    docs = [{"ctx": "q", "choices": [" a", " bb"], "gold": 1}]
    # raw: choice 0 wins (−1.2 > −1.5) → acc 0; per-char: −1.2/2 = −0.6 vs
    # −1.5/3 = −0.5 → choice 1 wins → acc_norm 1.
    scores = [ScoredItem(-1.2, 1, False), ScoredItem(-1.5, 2, False)]
    m = multiple_choice_metrics(docs, scores)
    assert m["n"] == 1
    assert m["acc"] == 0.0                       # −1.2 > −1.5 picks choice 0
    assert m["acc_norm"] == 1.0                  # −1.2/2 = −0.6 < −1.5/3 = −0.5 picks 1


def test_mc_doc_pairs_order():
    docs = [{"ctx": "c1", "choices": [" x", " y"], "gold": 0},
            {"ctx": "c2", "choices": [" z"], "gold": 0}]
    assert mc_doc_pairs(docs) == [("c1", " x"), ("c1", " y"), ("c2", " z")]


def test_lambada_metrics_math():
    scores = [ScoredItem(-2.0, 2, True), ScoredItem(-4.0, 2, False)]
    m = lambada_metrics(scores)
    assert m["acc"] == 0.5
    assert m["ppl"] == pytest.approx(math.exp(6.0 / 4.0), rel=1e-6)


def test_perplexity_uniform_and_window_edges():
    model = UniformModel()
    ids = list(range(3, 23))  # 20 tokens → windows of 8: [8, 8, 4]
    m = perplexity_over_tokens(model, ids, device="cpu", seq_len=8,
                               batch_size=2, pad_id=0)
    assert m["n_tokens"] == 20 - 3               # first token of each window unscored
    assert m["nll"] == pytest.approx(math.log(V), rel=1e-6)
    assert m["ppl"] == pytest.approx(V, rel=1e-4)
