"""In-process job registry for async ingest tasks.

Single-machine, single-process, no Redis. Sufficient for development and
small deployments running one ``uvicorn`` worker.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    status: str = JobStatus.PENDING
    stage: str = "queued"
    error: str | None = None
    doc_id: str | None = None
    num_chunks: int = 0
    filename: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobRegistry:
    """Thread-safe ``dict[str, Job]`` keyed by ``job_id``."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, **fields: Any) -> Job:
        job_id = uuid.uuid4().hex
        job = Job(job_id=job_id, **fields)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def update(self, job_id: str, **fields: Any) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for key, value in fields.items():
                if hasattr(job, key):
                    setattr(job, key, value)
                else:
                    job.extra[key] = value
            return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())
