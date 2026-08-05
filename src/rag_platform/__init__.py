"""RAG Platform — enterprise-grade production RAG system.

Modules:
- ``config``: environment-driven application settings.
- ``ingestion``: document loading, chunking, and PySpark pipelines.
- ``embeddings``: local SentenceTransformers embedding provider.
- ``vectorstore``: backend-agnostic vector store (Qdrant / Milvus).
- ``api``: FastAPI serving layer.
- ``evaluation``: Ragas-based RAG evaluation.
- ``mlflow_tracking``: experiment tracking helpers.
"""

__version__ = "0.1.0"
