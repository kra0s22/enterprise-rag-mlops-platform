"""Tunable client-side reciprocal rank fusion (RRF).

Qdrant's native ``FusionQuery(RRF)`` (server v1.12.x) uses a fixed RRF constant
and the client does not expose it, so this module provides a pure, tunable RRF
over ranked key lists. It is used for fusion sweeps (the RRF ``k`` and the dense/
sparse weights) and is fully unit-testable without a server.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def rrf_fuse(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[str]:
    """Fuse ``rankings`` with reciprocal rank fusion and return the fused order.

    Each item contributes ``weight / (k + rank)`` per list in which it appears,
    so items ranked high by several signals win. Higher ``k`` flattens the
    contribution of the top ranks; weights let one signal dominate.
    """
    if not rankings:
        return []
    if k <= 0:
        raise ValueError("k must be a positive integer")
    resolved_weights = weights if weights is not None else [1.0] * len(rankings)
    if len(resolved_weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, resolved_weights, strict=True):
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank + 1)

    return sorted(scores, key=scores.get, reverse=True)


def keys_from_hits(
    hits: Iterable[object], source_attr: str = "source", index_attr: str = "chunk_index"
) -> list[str]:
    """Build stable string keys for a sequence of payload-bearing hits."""
    keys: list[str] = []
    for hit in hits:
        payload = getattr(hit, "payload", {}) or {}
        source = payload.get(source_attr)
        index = payload.get(index_attr)
        if source is not None and index is not None:
            keys.append(f"{source}#{index}")
    return keys
