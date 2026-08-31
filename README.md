# RAG Platform

Enterprise-grade **Production RAG (Retrieval-Augmented Generation)** system built as a
portfolio showcase targeting European tech companies.

## Tech Stack

| Layer                 | Technology                                                           |
| --------------------- | -------------------------------------------------------------------- |
| Distributed ingestion | **PySpark** (distributed text processing & chunking)                 |
| Embeddings            | **SentenceTransformers** (local / OSS)                               |
| Generation            | **Ollama** (`llama3.2:3b`) — grounded answers from retrieved context |
| Vector Database       | **Qdrant** / **Milvus** (behind a common abstraction layer)          |
| Retrieval             | Dense + sparse **hybrid search** (RRF) + **cross-encoder reranking** |
| Serving API           | **FastAPI**                                                          |
| Evaluation            | **Ragas** (faithfulness, answer relevancy, context precision/recall) |
| Experiment tracking   | **MLflow**                                                           |
| Testing               | **pytest** (hermetic unit tests + opt-in integration suite)          |
| Deployment            | **Docker** + **docker-compose**                                      |

## Repository Layout

```text
.
├── .github/workflows/ci.yml     # CI: lint/tests, image build, on-demand eval
├── docker/                      # Dockerfiles + docker-compose
├── src/rag_platform/
│   ├── config/                  # Pydantic-settings configuration
│   ├── utils/                   # Logging helpers
│   ├── ingestion/               # Loader, token chunker, PySpark pipeline, CLI
│   ├── embeddings/              # Dense provider + hashing sparse encoder
│   ├── generation/              # Ollama grounded-generation client
│   ├── reranking/               # Cross-encoder reranking
│   ├── vectorstore/             # VectorStore abstraction + Qdrant/Milvus backends
│   ├── api/                     # FastAPI app (schemas, routes, dependencies)
│   ├── observability/           # Prometheus-style metrics collector
│   ├── evaluation/              # Ragas evaluation runner
│   └── mlflow_tracking/         # MLflow experiment tracking
├── tests/                       # pytest suites (unit, API, opt-in integration)
├── pyproject.toml               # Packaging, deps, lint/test config
└── .env.example                 # Environment variable reference
```

## Quickstart

```bash
# 1. Create a virtual environment (a repo-local conda env is used for development, see below)
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux

# 2. Install the package with dev extras
pip install -e ".[dev]"

# 3. Run the test suite
python -m pytest

# 4. Start Qdrant (Docker) and the API
docker compose -f docker/docker-compose.yml up -d qdrant
python -m uvicorn rag_platform.api.main:app --host 127.0.0.1 --port 8000
```

The API works out of the box against a local Qdrant at `http://localhost:6333`.
Copy `.env.example` to `.env` only to override any setting.

## Development environment

The project is developed on **Windows** with a **conda environment stored inside the
repository** at `.conda/` (gitignored). Dev tools and the console scripts
(`rag-api`, `rag-ingest`) are installed into `.conda\Scripts`, which is not on
`PATH` by default, so local commands call the environment's interpreter explicitly:

```powershell
# Lint / type check / tests
.conda\python.exe -m ruff check .
.conda\python.exe -m mypy src
.conda\python.exe -m pytest

# Run the API (docs at http://localhost:8000/docs)
.conda\python.exe -m uvicorn rag_platform.api.main:app --host 127.0.0.1 --port 8000

# Batch-ingest local documents into the vector store
.conda\python.exe -m rag_platform.ingestion.cli ./data/sample

# Distributed chunking with PySpark (requires a JDK reachable via JAVA_HOME)
.conda\python.exe -m rag_platform.ingestion.cli --distributed ./data/sample

# Start / stop the vector DB (Docker)
docker compose -f docker/docker-compose.yml up -d qdrant
docker compose -f docker/docker-compose.yml down

# Run the whole stack in Docker (API + Qdrant + MLflow). `--project-directory .`
# is required so the `api` build context resolves to the repository root.
docker compose -f docker/docker-compose.yml --project-directory . up -d --build
```

> If `.conda\Scripts` is added to `PATH`, the shorter `pytest`, `ruff`, `rag-api`
> and `rag-ingest` commands also work.

## Testing

The suite is split in two layers that complement each other:

- **Hermetic unit/API tests** (the default `pytest` run, also what CI executes): run
  offline with an in-memory Qdrant, a hash-based fake embedder and fake rerankers —
  deterministic and fast, they validate chunking, retrieval and API contracts without
  any model download, server or network.
- **Opt-in integration tests** (`tests/test_integration.py`): exercise the real stack
  (API + Qdrant + Ollama) end to end — health, dense/hybrid/reranked search and a
  grounded `/v1/rag` answer. They skip unless enabled, so CI stays deterministic:

```bash
# Hermetic suite (what CI runs)
python -m pytest

# Integration suite against a live stack (see docker/docker-compose.yml)
$env:RAG_RUN_INTEGRATION = "1"
python -m pytest tests/test_integration.py -v
```

## API Endpoints

| Method | Endpoint     | Description                                                 |
| ------ | ------------ | ----------------------------------------------------------- |
| `POST` | `/v1/ingest` | Chunk, embed, and store a document                          |
| `POST` | `/v1/search` | Retrieve the top-k nearest chunks for a query               |
| `POST` | `/v1/rag`    | Retrieve context and generate a grounded answer with Ollama |
| `GET`  | `/health`    | Liveness probe                                              |
| `GET`  | `/metrics`   | Prometheus metrics (requests, latency histogram)            |

### Grounded generation — `POST /v1/rag`

Retrieves the top-k relevant chunks for the question, injects them into a strict
*"answer only from context"* prompt, and calls a local **Ollama** model. Returns both
the answer and the `sources` it is grounded on, making every claim auditable:

```json
{
  "query": "how does qdrant store embeddings?",
  "answer": "Qdrant stores embeddings as high-dimensional vectors...",
  "sources": [
    {"id": "chunk-uuid", "score": 0.85, "chunk_text": "...", "metadata": {"source": "vector-db-notes"}}
  ]
}
```

If no relevant context is retrieved (or every hit is below `RAG_SCORE_THRESHOLD`),
the endpoint skips generation and returns a *"no relevant information"* response,
avoiding hallucinated answers.

## Environment Variables

All configuration is read from environment variables prefixed with `RAG_` (see `.env.example`).
Key variables:

- `RAG_VECTOR_STORE_PROVIDER` — `qdrant` or `milvus`
- `RAG_EMBEDDING_MODEL` — local SentenceTransformer model name
- `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` — token chunking window
- `RAG_CHUNK_MODE` — `window` (token) or `semantic` (structure-aware)
- `RAG_QDRANT_URL` / `RAG_MILVUS_URI` — vector DB endpoints
- `RAG_OLLAMA_URL` / `RAG_LLM_MODEL` — local Ollama endpoint and model for generation
- `RAG_LLM_TEMPERATURE` / `RAG_LLM_MAX_TOKENS` — generation sampling controls
- `RAG_SPARSE_DIM` — hashed sparse vector dimension used by hybrid retrieval
- `RAG_RERANKER_MODEL` / `RAG_RERANK_TOP_K` — cross-encoder model and candidate count for reranking
- `RAG_MLFLOW_TRACKING_URI` — MLflow server endpoint
- `RAG_API_KEY` — when set, `/v1/*` requires the `X-API-Key` header
- `RAG_RATE_LIMIT_PER_MINUTE` — per-client requests allowed each minute

## Hybrid search

Retrieval can fuse dense embeddings with hashing-based sparse vectors using
reciprocal rank fusion (RRF). Send `"hybrid": true` on `/v1/search` or `/v1/rag`;
documents are indexed with both representations automatically during ingestion.
`RAG_SPARSE_DIM` controls the sparse index size.

## Reranking

Sending `"rerank": true` on `/v1/search` or `/v1/rag` retrieves a wider candidate
set (`RAG_RERANK_TOP_K`, default 10) and re-scores it with a cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) before returning the final `top_k`,
lifting precision over the raw bi-encoder ranking. The pool can also be set per
request with `"rerank_top_k": N`; a sweep on the sample corpus found
`rerank_top_k=10` is the sweet spot (nDCG@5 0.988) — a pool of 5 loses recall
(0.967) while pools ≥10 return identical results.

## Query expansion (HyDE)

`"hyde": true` on `/v1/search` or `/v1/rag` generates a hypothetical passage for
the query with the local LLM and embeds that for retrieval instead of the raw
query. Measured on the sample corpus it did **not** improve over dense retrieval
(nDCG@5 0.93 vs 0.95): the small local model adds noise when queries are already
specific, so the flag stays opt-in.

## Evaluation

The Ragas evaluation stack is isolated in the `eval` extra to keep the serving image lean:

```bash
pip install -e ".[eval]"
```

With the API and a vector store running (see `docker/docker-compose.yml`), the end-to-end
runner queries `/v1/rag` for every question in the dataset, persists the collected samples,
and scores them with Ragas using the self-hosted Ollama LLM:

```bash
python -m rag_platform.evaluation.run_evaluation \
    --dataset ./data/eval_set.jsonl --api-url http://127.0.0.1:8000 --top-k 3
```

Append `--mlflow` to log the experiment parameters and metrics to the MLflow server
(`RAG_MLFLOW_TRACKING_URI`, experiment `rag_platform`):

```bash
python -m rag_platform.evaluation.run_evaluation \
    --dataset ./data/eval_set.jsonl --api-url http://127.0.0.1:8000 --top-k 3 --mlflow
```

The dataset is a JSONL file with `{question, ground_truth}`; the runner writes the collected
answers and retrieved contexts to `data/eval_results.jsonl` for reproducibility.

An on-demand CI job (`real-evaluation`) runs the same evaluation on every manual
dispatch when an `OPENAI_API_KEY` secret is configured, so model quality can be
tracked over time without a local LLM.

### Chunk-level retrieval metrics

The retrieval stage is also evaluated in isolation, without any LLM, so a gain can
be attributed to the store itself rather than to the generator. `run_retrieval_eval`
queries `/v1/search` for every query in `data/retrieval_set.jsonl` and scores the
ranked chunks with **MRR@k**, **hit-rate@k** and **nDCG@k**:

```bash
python -m rag_platform.evaluation.run_retrieval_eval \
    --dataset ./data/retrieval_set.jsonl --api-url http://127.0.0.1:8000 --top-k 5 \
    --hybrid --rerank --mlflow
```

On the sample corpus the three modes logged to MLflow show **hybrid+rerank
MRR@5 1.00 / nDCG@5 0.97**, dense 0.94/0.93, and hybrid alone 0.85/0.90 — the
sparse signal can hurt on a small corpus, but reranking recovers and beats both.

Chunking can also be ablated empirically: `run_ablation` re-ingests the corpus for
a grid of `chunk_size` × `chunk_overlap`, scores each configuration and logs it to
MLflow:

```bash
python -m rag_platform.evaluation.run_ablation \
    --chunk-sizes 200,300,400,512 --overlaps 32,64,128 --mlflow
```

On the sample corpus the optimum is **`chunk_size=300, chunk_overlap=64`
(nDCG@5 0.945)** — better than the default 512/64 (0.898) — while very large
overlaps hurt. The RRF constant is tunable client-side via `run_rrf_sweep`; on
this small corpus `k` has no measurable effect.

A structure-aware chunker is available via `RAG_CHUNK_MODE=semantic`: it groups
markdown sections instead of fixed token windows. Measured on the same corpus it
landed **below** the window optimum (nDCG@5 0.75 vs 0.95) — an honest result that
chunking strategy must be measured, not assumed.

A recent real run on the sample corpus (3 questions, `top_k 3`, judge `llama3.2:3b`,
dense retrieval) logged to MLflow produced **faithfulness 1.00**, **answer relevancy
0.70**, **context precision 0.92** and **context recall 0.73** on a clean
three-chunk collection, validating the retrieval + generation path end to end.

## Security & observability

When `RAG_API_KEY` is set, every `/v1/*` endpoint requires an `X-API-Key` header
(401 otherwise); `/health` and `/metrics` stay open. A per-client sliding-window
rate limiter caps requests at `RAG_RATE_LIMIT_PER_MINUTE` per minute (429 when
exceeded). A request middleware logs method/path/status/latency and feeds a
Prometheus-style collector exposed at `GET /metrics` — request counts by
method/route/status and a latency histogram, in Prometheus text format, with no
extra dependencies.

## Roadmap

Prioritised backlog for the retrieval and MLOps layers (effort: S/M/L).

### Production hardening

- Knowledge-base freshness and drift monitoring

### Scale & architecture

- Streaming ingestion (Spark structured streaming)
- Multi-tenancy (per-tenant partition isolation)
- Hybrid retrieval validation on Milvus (currently Qdrant-only)

### Ops

- Harden the serving base image (CVE pinning)
- Wire `OPENAI_API_KEY` secret for the on-demand `real-evaluation` CI job

## License

Proprietary. For portfolio demonstration purposes only.
