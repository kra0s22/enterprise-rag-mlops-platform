"""HyDE (Hypothetical Document Embeddings) query expansion.

Generates a hypothetical passage that would answer the query and embeds it for
retrieval. The passage tends to share vocabulary and structure with the indexed
documents, which can surface chunks the raw query would not match.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def hyde_query_vector(query: str, client: Any, embedder: Any) -> np.ndarray:
    """Return the embedding of a hypothetical passage answering ``query``.

    ``client`` is the generation client (provides ``generate_hypothesis``) and
    ``embedder`` the embedding provider used at query time.
    """
    passage = client.generate_hypothesis(query)
    return embedder.embed_query(passage)
