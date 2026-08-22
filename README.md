# RAG Platform

Enterprise-grade **Production RAG (Retrieval-Augmented Generation)** system built as a
portfolio showcase targeting European tech companies.

## Tech Stack

| Layer               | Technology                                                      |
| ------------------- | --------------------------------------------------------------- |
| Distributed ingestion| **PySpark** (distributed text processing & chunking)             |
| Embeddings          | **SentenceTransformers** (local / OSS)                           |
| Generation          | **Ollama** (`llama3.2:3b`) — grounded answers from retrieved context |
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
│   ├── generation/              # Ollama grounded-generation client
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

# Start / stop the vector DB (Docker)
docker compose -f docker/docker-compose.yml up -d qdrant
docker compose -f docker/docker-compose.yml down
```

> If `.conda\Scripts` is added to `PATH`, the shorter `pytest`, `ruff`, `rag-api`
> and `rag-ingest` commands also work.

## API Endpoints

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| `POST` | `/v1/ingest` | Chunk, embed, and store a document |
| `POST` | `/v1/search` | Retrieve the top-k nearest chunks for a query |
| `POST` | `/v1/rag` | Retrieve context and generate a grounded answer with Ollama |
| `GET`  | `/health` | Liveness probe |

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
- `RAG_QDRANT_URL` / `RAG_MILVUS_URI` — vector DB endpoints
- `RAG_OLLAMA_URL` / `RAG_LLM_MODEL` — local Ollama endpoint and model for generation
- `RAG_LLM_TEMPERATURE` / `RAG_LLM_MAX_TOKENS` — generation sampling controls
- `RAG_MLFLOW_TRACKING_URI` — MLflow server endpoint

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

## License

Proprietary. For portfolio demonstration purposes only.
