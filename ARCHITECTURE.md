# Architecture

## Overview

The platform is a modular, production-grade RAG system. Documents are ingested and
chunked in a distributed fashion with PySpark, embedded with a local OSS model, and
indexed into a vector database behind a common abstraction. A FastAPI service exposes
ingestion and retrieval endpoints (dense, hybrid dense+sparse, and cross-encoder
reranking), while Ragas and MLflow cover evaluation and experiment tracking.

```mermaid
flowchart LR
    subgraph Ingest
        DOC[Documents] --> LOAD[Loader]
        LOAD --> SPARK[PySpark Chunking]
        SPARK --> EMB[Embeddings]
    end
    EMB --> VS[(Vector DB\nQdrant / Milvus)]
    Q[Query] --> API[FastAPI]
    API --> EMB2[Embeddings]
    EMB2 --> VS
    VS --> API
    API --> RAG[RAG Response]
    subgraph Ops
        EVAL[Ragas] --> MLF[MLflow]
        SPARK --> MLF
        API --> MLF
    end
```

## Module Responsibilities

### `config/`

`Settings` is a `pydantic-settings` model loaded from `RAG_*` environment variables
(see `.env.example`). It centralizes every tunable knob: chunk size/overlap, embedding
model, vector DB endpoints, MLflow URI, and API binding. `get_settings()` is memoized
so the whole process shares one configuration object.

### `ingestion/`

- **`loader.py`** — loads `.txt`, `.md`, and `.pdf` documents into raw text.
- **`chunker.py`** — pure, dependency-light token chunking. A sliding window over
  whitespace tokens guarantees full coverage and overlapping boundaries, which is the
  main lever on retrieval recall/precision.
- **`spark_pipeline.py`** — expands a PySpark DataFrame of documents into chunk rows
  via a picklable UDF, enabling fully distributed chunking of large corpora.
- **`cli.py`** — `rag-ingest` batch command: loads files, chunks, embeds, and upserts
  into the configured vector store. `--distributed` chunks through `spark_pipeline.py`
  (a local PySpark session) instead of single-process tokenization.

### `embeddings/`

`EmbeddingProvider` is a small interface (`embed_documents`, `embed_query`,
`dimension`). `SentenceTransformerProvider` lazily loads a local
`sentence-transformers` model (torch is never imported on the serving import path),
normalizes embeddings, and batches encoding.

`HashingSparseEncoder` produces deterministic, L2-normalized sparse vectors via
feature hashing (sklearn). Queries and documents share the same vectorizer, so the
index space stays aligned for hybrid retrieval; it can be swapped for a SPLADE model
without changing call sites.

### `vectorstore/`

`VectorStore` is the abstraction contract: `create_collection`, `upsert`, `search`,
`delete`, `count`. `QdrantStore` and `MilvusStore` implement it. `build_vector_store`
selects the backend from configuration, so the rest of the platform is
backend-agnostic.

Points carry a named dense vector (`dense`) and, for hybrid backends, an optional
sparse vector (`sparse`). `supports_hybrid` gates `search_hybrid`, which Qdrant
implements with native reciprocal rank fusion (prefetch of both vectors +
`FusionQuery(RRF)`); Milvus stays dense-only and rejects sparse upserts.

### `reranking/`

`CrossEncoderReranker` scores `(query, candidate)` pairs with a
`sentence-transformers` cross-encoder, and `rerank_hits` reorders a retrieved list
while stamping the cross-encoder score. Endpoints fetch `rerank_top_k` candidates and
trim to `top_k` when `rerank` is enabled.

### `generation/`

`OllamaClient` builds a strict "answer only from context" prompt and calls a local
Ollama model, returning the grounded answer together with its retrieved sources.

### `api/`

FastAPI application with:

- `POST /v1/ingest` — ingest a document (text + metadata) → chunk → embed → upsert.
- `POST /v1/search` — embed a query and return top-k hits with metadata.
- `POST /v1/rag` — retrieve context and generate a grounded answer with Ollama.
- `GET /health` — liveness probe.

Requests accept `hybrid` (dense+sparse RRF) and `rerank` (cross-encoder) flags.
Dependencies (`get_embedder`, `get_vector_store`, `get_sparse_encoder`,
`get_reranker`) are singletons, created once per process and overridable in tests.

### `evaluation/`

`evaluate_rag` runs Ragas metrics (`faithfulness`, `answer_relevancy`,
`context_precision`, `context_recall`) over a dataset of
`{question, answer, contexts, ground_truth}` samples. Heavy Ragas imports are lazy so
the serving path stays lean. `run_evaluation` is an end-to-end runner: it queries
`/v1/rag` for every question, persists the collected samples, scores them with a
self-hosted LLM (Ollama) and optionally logs the run to MLflow. `--hybrid`/`--rerank`
switch the retrieval mode (dense by default), which is recorded as the `retrieval`
run parameter so retrieval variants can be compared (A/B) in MLflow.

`retrieval_metrics.py` provides pure, deterministic chunk-level metrics (`MRR@k`,
`hit-rate@k`, `nDCG@k`) over ranked lists of `(source, chunk_index)` keys, and
`run_retrieval_eval` scores the `/v1/search` ranking for a labelled query set,
allowing the retrieval stage to be tuned independently of the generator.

### `mlflow_tracking/`

`track_run` is a context manager that opens an MLflow run and logs parameters; metrics
and artifacts are logged inside the block. Used to track ingestion pipelines and
evaluation experiments.

## Design Decisions

- **Lazy heavy imports** — torch (via sentence-transformers), Ragas, and MLflow are
  imported only where used, so the API image and import time stay minimal.
- **Backend-agnostic vector store** — the `VectorStore` interface means Qdrant and
  Milvus are drop-in replacements behind a single factory.
- **Deterministic, hermetic tests** — the test suite uses a fake embedding provider and
  Qdrant in-memory mode (`path=":memory:"`), so tests run without a running DB or
  network access.
- **Overlapping token chunks** — overlap mitigates context loss at chunk boundaries,
  improving retrieval quality for downstream RAG.
- **Native hybrid fusion (RRF)** — Qdrant's prefetch + `FusionQuery(RRF)` merges dense
  and sparse rankings server-side, with no client-side score arithmetic.
- **Hashing sparse encoder** — dependency-free, deterministic sparse vectors; a SPLADE
  model can replace it behind the same interface.
- **Reranking as an opt-in flag** — a wider candidate set is re-scored by a
  cross-encoder only when requested, keeping default latency minimal.
- **Measurable retrieval A/B** — the evaluation runner can target dense or
  hybrid+rerank retrieval; on the sample corpus this showed `context_recall`
  improving from 0.83 to 0.97, a quantitative proof of the retrieval stack.
