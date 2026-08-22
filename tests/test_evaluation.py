"""Hermetic tests for the Ragas evaluation helpers and the evaluation runner.

The module is skipped when ``ragas`` is not installed, so CI (which installs only
the ``dev`` extra) stays green while local runs exercise the tests.
"""

from __future__ import annotations

import pytest

from rag_platform.evaluation import ragas_eval
from rag_platform.evaluation.run_evaluation import collect_answers

ragas = pytest.importorskip("ragas")


class _Series:
    """Minimal stand-in for a pandas Series (only ``mean`` is needed)."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def mean(self) -> float:
        return sum(self._values) / len(self._values)


class _Frame:
    """Minimal stand-in for the DataFrame returned by ``to_pandas``."""

    def __init__(self, values: dict[str, list[float]]) -> None:
        self._values = values

    def __getitem__(self, name: str) -> _Series:
        return _Series(self._values[name])


class _FakeResult:
    """Minimal stand-in for a Ragas evaluation result."""

    def __init__(self, values: dict[str, float]) -> None:
        self._frame = _Frame({name: [value] for name, value in values.items()})

    def to_pandas(self) -> _Frame:
        return self._frame


def test_evaluate_rag_returns_metric_scores(monkeypatch) -> None:
    def fake_evaluate(*, dataset, metrics, llm=None, embeddings=None) -> _FakeResult:
        assert len(dataset.samples) == 1
        assert llm == "fake-llm"
        assert embeddings == "fake-embeddings"
        return _FakeResult({"faithfulness": 0.9, "answer_relevancy": 0.8})

    monkeypatch.setattr(ragas, "evaluate", fake_evaluate)
    scores = ragas_eval.evaluate_rag(
        [{"question": "q", "answer": "a", "contexts": ["ctx"]}],
        metrics=["faithfulness", "answer_relevancy"],
        llm="fake-llm",
        embeddings="fake-embeddings",
    )
    assert scores == {"faithfulness": 0.9, "answer_relevancy": 0.8}


def test_evaluate_rag_rejects_unknown_metrics() -> None:
    try:
        ragas_eval.evaluate_rag(
            [{"question": "q", "answer": "a", "contexts": ["ctx"]}], metrics=["nope"]
        )
    except ValueError as exc:
        assert "No valid metrics" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown metric")


class _FakeResponse:
    def __init__(self, body: dict) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._body


class _FakeClient:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def post(self, url: str, json: dict) -> _FakeResponse:
        self.calls.append((url, json))
        return _FakeResponse(self._responses.pop(0))


def test_collect_answers_extracts_contexts_and_ground_truth(monkeypatch) -> None:
    import rag_platform.evaluation.run_evaluation as run_eval

    responses = [
        {"answer": "a1", "sources": [{"chunk_text": "ctx1", "score": 0.9}]},
        {"answer": "a2", "sources": []},
    ]
    client = _FakeClient(responses)
    monkeypatch.setattr(run_eval.httpx, "Client", lambda **_: client)

    dataset = [{"question": "q1"}, {"question": "q2", "ground_truth": "gt2"}]
    samples = collect_answers(dataset, "http://fake", 3)

    assert samples[0]["contexts"] == ["ctx1"]
    assert samples[1]["contexts"] == []
    assert samples[1]["ground_truth"] == "gt2"
    assert client.calls == [
        ("/v1/rag", {"query": "q1", "top_k": 3}),
        ("/v1/rag", {"query": "q2", "top_k": 3}),
    ]
