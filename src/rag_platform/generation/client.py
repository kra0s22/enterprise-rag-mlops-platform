"""Grounded generation client backed by a local Ollama model.

The retrieved chunks are injected into a strict "answer only from context"
prompt, which is the main lever against hallucination in the RAG loop.
"""

from __future__ import annotations

from typing import Any

import httpx

from rag_platform.utils.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer ONLY using the provided context. "
    "If the context does not contain the answer, say that you do not know. "
    "Do not invent facts."
)


def build_grounded_prompt(query: str, contexts: list[str]) -> str:
    """Assemble the user prompt with the retrieved chunks as grounding context."""
    context_block = "\n---\n".join(contexts)
    return (
        f"Context (from the knowledge base):\n---\n{context_block}\n---\n\n"
        f"Question: {query}\n\nAnswer:"
    )


class OllamaClient:
    """Thin client over the local Ollama ``/api/generate`` endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, query: str, contexts: list[str]) -> str:
        """Return a grounded answer for ``query`` given the retrieved ``contexts``."""
        prompt = build_grounded_prompt(query, contexts)
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "system": _SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
        logger.info(
            "Generating answer for query=%r with %d context chunks", query, len(contexts)
        )
        response = httpx.post(
            f"{self._base_url}/api/generate", json=payload, timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]


def build_generation_client(settings: Any) -> OllamaClient:
    """Instantiate the generation client from application settings."""
    return OllamaClient(
        base_url=settings.ollama_url,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
