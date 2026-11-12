"""
celery_app.py
-------------
Celery app for background upload reading (adapted from text-extract-api, MIT).

Run a worker (from the project root):
    celery -A agent.uploads.celery_app worker --loglevel=info --concurrency=2

Broker + results: Redis (settings.json -> uploads.redis_url, or REDIS_URL env var).
"""

from __future__ import annotations

import os

from celery import Celery

from agent.uploads.files import upload_settings

_cfg = upload_settings()
REDIS_URL = os.getenv("REDIS_URL") or _cfg["redis_url"]

app = Celery("helper_quote", broker=REDIS_URL, backend=REDIS_URL, include=["agent.uploads.tasks"])
app.conf.update(
    task_acks_late=True,                     # a crashed worker's job is re-delivered
    worker_prefetch_multiplier=1,            # one long job at a time per worker process
    task_time_limit=int(_cfg["task_time_limit_seconds"]) + 30,
    task_soft_time_limit=int(_cfg["task_time_limit_seconds"]),
    task_track_started=True,
    result_expires=24 * 3600,
    worker_max_memory_per_child=600_000,     # KB - recycle a worker that grew too big
    broker_connection_retry_on_startup=True,
)
