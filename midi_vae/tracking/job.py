"""Job management for tracking individual pipeline stage executions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from midi_vae.utils.io import save_json, load_json


@dataclass
class Job:
    """Tracks execution of a single pipeline stage or sub-task.

    Records timing, status, and any error information.
    """

    name: str
    status: str = "pending"  # pending | running | completed | failed
    start_time: float | None = None
    end_time: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        """Mark job as running."""
        self.status = "running"
        self.start_time = time.time()

    def complete(self) -> None:
        """Mark job as completed."""
        self.status = "completed"
        self.end_time = time.time()

    def fail(self, error: str) -> None:
        """Mark job as failed.

        Args:
            error: Error message or traceback.
        """
        self.status = "failed"
        self.end_time = time.time()
        self.error = error

    @property
    def elapsed(self) -> float | None:
        """Elapsed time in seconds, or None if not started."""
        if self.start_time is None:
            return None
        end = self.end_time or time.time()
        return end - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "name": self.name,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_seconds": self.elapsed,
            "error": self.error,
            "metadata": self.metadata,
        }


class JobManager:
    """Manages a collection of jobs for an experiment.

    Provides methods to create, track, and persist job records.
    """

    def __init__(self, jobs_dir: Path) -> None:
        """Initialize the job manager.

        Args:
            jobs_dir: Directory to save job records.
        """
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}

    def create_job(self, name: str) -> Job:
        """Create and register a new job.

        Args:
            name: Unique job name.

        Returns:
            The created Job instance.
        """
        job = Job(name=name)
        self.jobs[name] = job
        return job

    def save_job(self, job: Job) -> None:
        """Save job record to disk.

        Args:
            job: Job to save.
        """
        save_json(job.to_dict(), self.jobs_dir / f"{job.name}.json")

    def load_jobs(self) -> dict[str, Job]:
        """Load all job records from disk.

        Returns:
            Dict of job name -> Job.
        """
        jobs: dict[str, Job] = {}
        for f in self.jobs_dir.glob("*.json"):
            data = load_json(f)
            job = Job(
                name=data["name"],
                status=data.get("status", "unknown"),
                start_time=data.get("start_time"),
                end_time=data.get("end_time"),
                error=data.get("error"),
                metadata=data.get("metadata", {}),
            )
            jobs[job.name] = job
        self.jobs = jobs
        return jobs
