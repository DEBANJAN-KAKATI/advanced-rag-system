"""
Retrieval metrics computed against chunking-independent evidence spans.

For each question we know which evidence spans answer it. A retrieved chunk is
relevant if it contains one of those spans (see dataset.chunk_covers). From that:

  hit@k              any relevant chunk in the top k
  evidence_recall@k  fraction of required spans present in the top k
                     (for evidence_mode="any", 1 if any one span is present)
  mrr                1 / rank of the first relevant chunk (0 if none in the list)
  ndcg@k             binary-relevance nDCG, ideal ranking computed from every
                     relevant chunk in the index
  precision@k        fraction of the top k chunks that are relevant
  context_tokens@k   tokens the LLM would read if given the top k chunks
  recall@budget      evidence recall when chunks are packed, in rank order,
                     into a fixed token budget -- the fair way to compare
                     chunk sizes, since bigger chunks trivially raise recall@k
"""
import math
from typing import Dict, List, Sequence

import numpy as np

from app.core.pricing import estimate_tokens
from evaluation.dataset import QAItem, chunk_covers

KS = (1, 3, 5, 10)
TOKEN_BUDGETS = (500, 1000, 2000)


def coverage_matrix(item: QAItem, chunks: Sequence[dict]) -> List[List[bool]]:
    return [[chunk_covers(c, ev) for ev in item.evidence] for c in chunks]


def _evidence_recall(cover: List[List[bool]], n_spans: int, mode: str) -> float:
    found = [any(row[j] for row in cover) for j in range(n_spans)]
    if mode == "any":
        return float(any(found))
    return sum(found) / n_spans


def retrieval_metrics(item: QAItem, ranked: Sequence[dict], n_relevant_in_index: int) -> Dict[str, float]:
    cover = coverage_matrix(item, ranked)
    relevant = [any(row) for row in cover]
    n = len(item.evidence)
    out: Dict[str, float] = {}
    for k in KS:
        out[f"hit@{k}"] = float(any(relevant[:k]))
        out[f"evidence_recall@{k}"] = _evidence_recall(cover[:k], n, item.evidence_mode)
    out["precision@5"] = sum(relevant[:5]) / 5
    first = next((i for i, r in enumerate(relevant) if r), None)
    out["mrr"] = 0.0 if first is None else 1.0 / (first + 1)

    k = max(KS)
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(relevant[:k]) if r)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, n_relevant_in_index)))
    out[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0

    tokens = [estimate_tokens(c["text"]) for c in ranked]
    out["context_tokens@5"] = float(sum(tokens[:5]))
    for budget in TOKEN_BUDGETS:
        used, packed = 0, 0
        for t in tokens:
            if used + t > budget:
                break
            used += t
            packed += 1
        out[f"recall@{budget}tok"] = _evidence_recall(cover[:packed], n, item.evidence_mode) if packed else 0.0
    return out


def max_achievable_recall(item: QAItem, all_chunks: Sequence[dict]) -> float:
    """Upper bound on evidence recall for this chunking: can any chunk hold each span?"""
    cover = coverage_matrix(item, all_chunks)
    return _evidence_recall(cover, len(item.evidence), item.evidence_mode)


def percentile(values: List[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def mean(values: List[float]) -> float:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_ci(values: List[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """95% percentile-bootstrap confidence interval of the mean."""
    vals = np.array([v for v in values if v is not None], dtype=float)
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_bootstrap_pvalue(a: List[float], b: List[float], n_boot: int = 5000, seed: int = 0) -> float:
    """Two-sided p-value that mean(a) == mean(b) for per-question paired scores."""
    diff = np.array(a, dtype=float) - np.array(b, dtype=float)
    observed = diff.mean()
    rng = np.random.default_rng(seed)
    centered = diff - observed
    boots = rng.choice(centered, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return float((np.abs(boots) >= abs(observed)).mean())
