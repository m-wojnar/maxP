"""Downstream-eval scoring library.

Pure scoring: no torchtitan imports, so the logic is unit-testable on CPU
(tests/test_eval_tasks.py). Dataset loading imports `datasets` lazily inside
the task builders; the scoring core needs only torch.

Conventions (fixed here, identical across all arms — comparisons are relative):
- A "doc" is {"ctx": str, "choices": [str, ...], "gold": int}. Choice strings
  carry their leading space.
- Continuation token ids come from re-encoding the whole string:
  cont_ids = enc(ctx + choice)[len(enc(ctx)):], the standard way to handle
  tokenizer boundary merges.
- Scores are computed from float32 log-softmax over bf16 logits.
- Batches are padded to a FIXED (batch, length) shape so a compiled model sees
  one graph per task family. Padding is on the right; causal attention makes
  right-padding inert for the scored positions.
- acc: argmax of total continuation log-prob. acc_norm: argmax of
  log-prob / len(choice string in chars).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# tokenizer adapter
# --------------------------------------------------------------------------- #

def encode_plain(tok, text: str) -> list[int]:
    """Encode without BOS/EOS regardless of the tokenizer's auto-add settings."""
    try:
        ids = tok.encode(text, add_bos=False, add_eos=False)
    except TypeError:
        ids = tok.encode(text)
        bos = getattr(tok, "bos_id", None)
        eos = getattr(tok, "eos_id", None)
        if bos is not None and ids and ids[0] == bos:
            ids = ids[1:]
        if eos is not None and ids and ids[-1] == eos:
            ids = ids[:-1]
    return list(ids)


def split_context_continuation(tok, ctx: str, cont: str) -> tuple[list[int], list[int]]:
    """Token ids for ctx and for cont, with boundary merges resolved by
    re-encoding the concatenation. Guarantees len(cont_ids) >= 1."""
    ctx_ids = encode_plain(tok, ctx)
    full_ids = encode_plain(tok, ctx + cont)
    n = len(ctx_ids)
    # A boundary merge can swallow the last ctx token; back off until the
    # prefix matches, so the continuation always begins at a real boundary.
    while n > 0 and full_ids[:n] != ctx_ids[:n]:
        n -= 1
    cont_ids = full_ids[n:]
    if not cont_ids:  # pathological (cont whitespace merged away): score last token
        n -= 1
        cont_ids = full_ids[n:]
    return full_ids[:n], cont_ids


# --------------------------------------------------------------------------- #
# scoring core
# --------------------------------------------------------------------------- #

@dataclass
class ScoredItem:
    logprob: float     # sum of continuation-token log-probs
    n_tokens: int
    greedy: bool       # every continuation token is the argmax at its position


@torch.no_grad()
def score_batch(model, requests: list[tuple[list[int], list[int]]],
                pad_id: int, seq_len: int, device) -> list[ScoredItem]:
    """Score (ctx_ids, cont_ids) pairs in one fixed-shape forward.

    Each row is [ctx cont pad...] of exactly seq_len tokens. Logits at position
    p predict token p+1, so continuation tokens at absolute positions
    [c, c+k) are read from logits[c-1 : c+k-1]. Rows must satisfy
    len(ctx)+len(cont) <= seq_len and len(ctx) >= 1.
    """
    B = len(requests)
    toks = torch.full((B, seq_len), pad_id, dtype=torch.long)
    spans = []
    for i, (ctx_ids, cont_ids) in enumerate(requests):
        if not ctx_ids:
            raise ValueError("empty context (BOS should guarantee >= 1 token)")
        total = len(ctx_ids) + len(cont_ids)
        if total > seq_len:  # truncate context from the LEFT, keep continuation
            drop = total - seq_len
            ctx_ids = ctx_ids[drop:]
        row = ctx_ids + cont_ids
        toks[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        spans.append((len(ctx_ids), len(cont_ids)))
    toks = toks.to(device)
    logits = model(toks)
    if isinstance(logits, tuple):
        logits = logits[0]
    logp = F.log_softmax(logits.float(), dim=-1)

    out = []
    for i, (c, k) in enumerate(spans):
        rows = logp[i, c - 1 : c + k - 1]                       # [k, V]
        targets = toks[i, c : c + k]                            # [k]
        token_lp = rows.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        greedy = bool((rows.argmax(dim=-1) == targets).all().item())
        out.append(ScoredItem(float(token_lp.sum().item()), k, greedy))
    return out


def score_requests(model, tok, pairs: list[tuple[str, str]], *, device,
                   batch_size: int, seq_len: int, bos: bool = True) -> list[ScoredItem]:
    """Tokenize and score (ctx, cont) string pairs with fixed-shape batches.

    The final partial batch is padded with copies of its first request; the
    padded rows are discarded, so results align 1:1 with `pairs`.
    """
    pad_id = getattr(tok, "eos_id", None) or 0
    bos_id = getattr(tok, "bos_id", None)
    reqs = []
    for ctx, cont in pairs:
        ctx_ids, cont_ids = split_context_continuation(tok, ctx, cont)
        if bos and bos_id is not None:
            ctx_ids = [bos_id] + ctx_ids
        reqs.append((ctx_ids, cont_ids))
    results: list[ScoredItem] = []
    for start in range(0, len(reqs), batch_size):
        chunk = reqs[start : start + batch_size]
        n_real = len(chunk)
        while len(chunk) < batch_size:
            chunk.append(chunk[0])
        results.extend(score_batch(model, chunk, pad_id, seq_len, device)[:n_real])
    return results


# --------------------------------------------------------------------------- #
# metrics over docs
# --------------------------------------------------------------------------- #

def multiple_choice_metrics(docs: list[dict], scores: list[ScoredItem]) -> dict:
    """acc / acc_norm over docs; `scores` is flat, one entry per (doc, choice)."""
    n = len(docs)
    correct = correct_norm = 0
    i = 0
    for doc in docs:
        k = len(doc["choices"])
        lps = [scores[i + j].logprob for j in range(k)]
        lens = [max(1, len(doc["choices"][j])) for j in range(k)]
        pick = max(range(k), key=lambda j: lps[j])
        pick_norm = max(range(k), key=lambda j: lps[j] / lens[j])
        correct += pick == doc["gold"]
        correct_norm += pick_norm == doc["gold"]
        i += k
    return {"acc": correct / n, "acc_norm": correct_norm / n, "n": n}


def lambada_metrics(scores: list[ScoredItem]) -> dict:
    n = len(scores)
    acc = sum(s.greedy for s in scores) / n
    total_lp = sum(s.logprob for s in scores)
    total_tok = sum(s.n_tokens for s in scores)
    return {"acc": acc, "ppl": float(torch.tensor(-total_lp / total_tok).exp()), "n": n}


@torch.no_grad()
def perplexity_over_tokens(model, ids: list[int], *, device, seq_len: int,
                           batch_size: int, pad_id: int) -> dict:
    """Token-level ppl over a corpus, disjoint windows of seq_len (stride =
    seq_len). Within a window every token after the first is scored; window-
    initial tokens are unscored (no context). Deterministic, convention fixed."""
    windows = [ids[i : i + seq_len] for i in range(0, len(ids), seq_len)]
    if len(windows[-1]) < 2:
        windows = windows[:-1]
    total_lp, total_tok = 0.0, 0
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        n_real = len(chunk)
        while len(chunk) < batch_size:
            chunk.append(chunk[0])
        toks = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
        lens = []
        for i, w in enumerate(chunk):
            toks[i, : len(w)] = torch.tensor(w, dtype=torch.long)
            lens.append(len(w))
        toks = toks.to(device)
        logits = model(toks)
        if isinstance(logits, tuple):
            logits = logits[0]
        logp = F.log_softmax(logits.float(), dim=-1)
        for i in range(n_real):
            L = lens[i]
            rows = logp[i, : L - 1]
            targets = toks[i, 1:L]
            total_lp += float(rows.gather(-1, targets.unsqueeze(-1)).sum().item())
            total_tok += L - 1
    return {"ppl": float(torch.tensor(-total_lp / total_tok).exp()),
            "nll": -total_lp / total_tok, "n_tokens": total_tok}


# --------------------------------------------------------------------------- #
# task builders (datasets imported lazily; templates FIXED — do not edit
# between arms, absolute values are not comparable to lm-eval-harness)
# --------------------------------------------------------------------------- #

def build_hellaswag(limit: int | None = None) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("hellaswag", split="validation")
    docs = []
    for ex in ds:
        ctx = ex["activity_label"] + ": " + ex["ctx_a"]
        if ex["ctx_b"]:
            ctx += " " + ex["ctx_b"].capitalize()
        docs.append({"ctx": ctx,
                     "choices": [" " + e for e in ex["endings"]],
                     "gold": int(ex["label"])})
        if limit and len(docs) >= limit:
            break
    return docs


def build_arc_easy(limit: int | None = None) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("ai2_arc", "ARC-Easy", split="test")
    docs = []
    for ex in ds:
        labels = ex["choices"]["label"]
        if ex["answerKey"] not in labels:
            continue
        docs.append({"ctx": "Question: " + ex["question"] + "\nAnswer:",
                     "choices": [" " + t for t in ex["choices"]["text"]],
                     "gold": labels.index(ex["answerKey"])})
        if limit and len(docs) >= limit:
            break
    return docs


def build_piqa(limit: int | None = None) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("ybisk/piqa", split="validation",
                      revision="refs/convert/parquet")
    docs = []
    for ex in ds:
        docs.append({"ctx": "Question: " + ex["goal"] + "\nAnswer:",
                     "choices": [" " + ex["sol1"], " " + ex["sol2"]],
                     "gold": int(ex["label"])})
        if limit and len(docs) >= limit:
            break
    return docs


def build_lambada(limit: int | None = None) -> list[tuple[str, str]]:
    """(ctx, last_word) pairs; the continuation is ' ' + final whitespace-word."""
    from datasets import load_dataset
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    pairs = []
    for ex in ds:
        text = ex["text"].rstrip()
        head, _, last = text.rpartition(" ")
        if not head:
            continue
        pairs.append((head, " " + last))
        if limit and len(pairs) >= limit:
            break
    return pairs


def build_wikitext_ids(tok, limit_tokens: int | None = None) -> list[int]:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    text = "".join(ex["text"] for ex in ds)
    ids = encode_plain(tok, text)
    return ids[:limit_tokens] if limit_tokens else ids


def mc_doc_pairs(docs: list[dict]) -> list[tuple[str, str]]:
    """Flatten docs into (ctx, choice) request pairs, doc-major order."""
    return [(d["ctx"], ch) for d in docs for ch in d["choices"]]
