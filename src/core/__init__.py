"""Transport-agnostic services for ParaCite.

These classes are shared by the HTTP API (``app.api``) and the CLI
(``app.cli``). They never touch transport details (HTTP requests, argparse,
etc.) and never load FAISS / NLTK / sentence-transformers themselves -- those
are injected by the caller (see ``app.__init__`` for the wiring).
"""

from src.core.ask_service import AskService
from src.core.ingest_service import IngestService
from src.core.jobs import Job, JobRegistry, JobStatus

__all__ = ["AskService", "IngestService", "Job", "JobRegistry", "JobStatus"]
