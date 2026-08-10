# RAG Platform

Enterprise-grade **Production RAG (Retrieval-Augmented Generation)** system built as a
portfolio showcase targeting European tech companies.

## Tech Stack

| Layer               | Technology                                                      |
| ------------------- | --------------------------------------------------------------- |
| Distributed ingestion| **PySpark** (distributed text processing & chunking)             |
| Embeddings          | **SentenceTransformers** (local / OSS)                           |
| Vector Database     | **Qdrant** / **Milvus** (behind a common abstraction layer)      |
| Serving API         | **FastAPI**                                                      |
| Evaluation          | **Ragas** (faithfulness, answer relevancy, context precision/recall) |
| Experiment tracking | **MLflow**                                                       |
| Testing             | **pytest**                                                       |
| Deployment          | **Docker** + **docker-compose**                                  |

## Repository Layout

```
.
├── .github/workflows/ci.yml     # CI pipeline (lint + tests)
├── docker/                      # Dockerfiles + docker-compose
├── src/rag_platform/
│   ├── config/                  # Pydantic-settings configuration
│   ├── utils/                   # Logging helpers
│   ├── ingestion/               # Loader, token chunker, PySpark pipeline, CLI
│   ├── embeddings/              # SentenceTransformers provider
│   ├── vectorstore/             # VectorStore abstraction + Qdrant/Milvus backends
│   ├── api/                     # FastAPI app (schemas, routes, dependencies)
│   ├── evaluation/              # Ragas evaluation runner
│   └── mlflow_tracking/         # MLflow experiment tracking
├── tests/                       # pytest suites (chunking, retrieval, API)
├── pyproject.toml               # Packaging, deps, lint/test config
└── .env.example                 # Environment variable reference
```

## Quickstart

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install the package with dev extras
pip install -e ".[dev]"

# 3. Copy and adjust environment configuration
cp .env.example .env

# 4. Run the test suite
pytest

# 5. Start Qdrant (Docker) and the API
docker compose up -d qdrant
rag-api
```

## Core Commands

```bash
# Lint
ruff check .

# Type check
mypy src

# Run the API (http://localhost:8000/docs)
rag-api

# Batch-ingest local documents into the vector store
rag-ingest ./data/sample
```

## Environment Variables

All configuration is read from environment variables prefixed with `RAG_` (see `.env.example`).
Key variables:

- `RAG_VECTOR_STORE_PROVIDER` — `qdrant` or `milvus`
- `RAG_EMBEDDING_MODEL` — local SentenceTransformer model name
- `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` — token chunking window
- `RAG_QDRANT_URL` / `RAG_MILVUS_URI` — vector DB endpoints
- `RAG_MLFLOW_TRACKING_URI` — MLflow server endpoint

## Evaluation

The Ragas evaluation stack is isolated in the `eval` extra to keep the serving image lean:

```bash
pip install -e ".[eval]"
python -m rag_platform.evaluation.ragas_eval --dataset ./data/eval_set.jsonl
```

## License

Proprietary. For portfolio demonstration purposes only.
