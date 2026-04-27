"""FastAPI app exposing the two ParaCite endpoints.

- ``POST /ingest`` - async file upload, returns a ``job_id``
- ``GET  /ingest/{job_id}`` - poll status
- ``POST /ask`` - synchronous claim extraction + retrieval
- ``GET  /health`` - liveness + index size
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.app import get_ask_service, get_ingest_service, get_jobs, get_store
from src.config import UPLOADS_RAW_DIR
from src.core import JobStatus
from src.ingest.extract_text import SUPPORTED_SUFFIXES, UnsupportedFileType

log = logging.getLogger(__name__)

app = FastAPI(title="ParaCite", version="0.1.0")


class AskRequest(BaseModel):
    text: str = Field(..., description="User text to find citable claims in")
    top_k: int = Field(3, ge=1, le=50)
    style: str = Field("apa", description="Citation style (currently passthrough)")


class IngestStartResponse(BaseModel):
    job_id: str
    status: str
    filename: str


@app.on_event("startup")
def _warm_singletons() -> None:
    """Load embedder + index eagerly so the first request isn't slow."""
    get_store()
    get_ask_service()
    get_ingest_service()


@app.get("/health")
def health() -> dict[str, Any]:
    store = get_store()
    return {
        "status": "ok",
        "index_size": int(store.index.ntotal) if store else 0,
        "num_records": len(store.records) if store else 0,
    }


@app.post("/ingest", response_model=IngestStartResponse, status_code=202)
async def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> IngestStartResponse:
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type {suffix!r}; supported: {sorted(SUPPORTED_SUFFIXES)}",
        )

    UPLOADS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    jobs = get_jobs()
    job = jobs.create(filename=filename, status=JobStatus.PENDING, stage="uploaded")
    saved_path = UPLOADS_RAW_DIR / f"{job.job_id}{suffix}"
    try:
        with saved_path.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
    finally:
        await file.close()

    background_tasks.add_task(_run_ingest_job, job.job_id, saved_path)

    return IngestStartResponse(job_id=job.job_id, status=job.status, filename=filename)


@app.get("/ingest/{job_id}")
def ingest_status(job_id: str) -> dict[str, Any]:
    job = get_jobs().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job_id: {job_id}")
    return job.to_dict()


@app.post("/ask")
def ask(payload: AskRequest) -> dict[str, Any]:
    return get_ask_service().ask(
        payload.text, top_k=payload.top_k, style=payload.style,
    )


def _run_ingest_job(job_id: str, path: Path) -> None:
    jobs = get_jobs()
    jobs.update(job_id, status=JobStatus.RUNNING, stage="starting")

    def on_progress(stage: str, info: dict[str, Any]) -> None:
        jobs.update(job_id, stage=stage, **info)

    try:
        result = get_ingest_service().ingest(path, on_progress=on_progress)
        jobs.update(
            job_id,
            status=JobStatus.DONE,
            stage="done",
            doc_id=result["doc_id"],
            num_chunks=result["num_chunks"],
        )
    except UnsupportedFileType as exc:
        jobs.update(job_id, status=JobStatus.FAILED, error=str(exc), stage="failed")
    except Exception as exc:  # noqa: BLE001 - top-level worker boundary
        log.exception("ingest job %s failed", job_id)
        jobs.update(job_id, status=JobStatus.FAILED, error=str(exc), stage="failed")
