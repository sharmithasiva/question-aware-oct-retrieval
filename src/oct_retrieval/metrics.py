"""Ranking metrics with binary relevance."""
from __future__ import annotations

import math
from collections.abc import Sequence


def hit_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    return float(any(r in relevant for r in ranked[:k]))


def recall_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    return len(relevant.intersection(ranked[:k])) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: Sequence, relevant: set) -> float:
    for i, r in enumerate(ranked, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(ranked[:k]) if r in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0
