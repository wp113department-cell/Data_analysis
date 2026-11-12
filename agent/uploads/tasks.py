"""
tasks.py
--------
The background job: prepare every page of a batch, read it with the vision model,
cache the reading by file hash, and save the result for the agent.

Task structure (bind=True + PROGRESS states + Redis cache by file hash) adapted
from text-extract-api (CatchTheTornado, MIT).
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from agent.uploads.celery_app import REDIS_URL, app
from agent.uploads.files import load_batch, prepare_pages, result_path
from agent.uploads.vision import read_pages

CACHE_PREFIX = "hq:vision:v1:"
CACHE_SECONDS = 30 * 24 * 3600


def _redis():
    try:
        import redis

        client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=1, socket_timeout=3)
        client.ping()
        return client
    except Exception:  # noqa: BLE001 - no cache without Redis; reading still works
        return None


def write_status(thread_id: str, batch_id: str, **fields: Any) -> None:
    path = result_path(thread_id, batch_id).with_name("status.json")
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(fields, updated_at=datetime.now().isoformat(timespec="seconds"))
    path.write_text(json.dumps(current))


def process_batch(thread_id: str, batch_id: str,
                  on_progress: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
    """Read every file of a batch. Used by the Celery task and by the in-app thread fallback."""
    started = time.time()

    def report(pct: int, status: str) -> None:
        write_status(thread_id, batch_id, state="running", progress=pct, status=status)
        if on_progress:
            on_progress(pct, status)

    batch = load_batch(thread_id, batch_id)
    if batch is None:
        raise FileNotFoundError(f"Upload batch {batch_id} not found")
    cache = _redis()
    total_pages = sum(f.pages for f in batch.files) or 1
    pages_done = 0
    files_out: List[Dict[str, Any]] = []
    report(5, "Preparing your files")

    for f in batch.files:
        cached = cache.get(CACHE_PREFIX + f.sha256) if cache else None
        if cached:
            readings = json.loads(cached)
            pages_done += f.pages
            report(10 + int(85 * pages_done / total_pages), f"{f.name}: already read before (cached)")
        else:
            pages = prepare_pages(f)
            base = pages_done

            def page_progress(done: int, total: int, label: str) -> None:
                report(10 + int(85 * (base + done) / total_pages), label)

            readings = read_pages(pages, progress=page_progress)
            pages_done += len(pages)
            if cache and not any(r.get("error") for r in readings):
                cache.setex(CACHE_PREFIX + f.sha256, CACHE_SECONDS, json.dumps(readings))
        files_out.append({"name": f.name, "kind": f.kind, "pages": readings})

    result = {
        "batch_id": batch_id,
        "thread_id": thread_id,
        "files": files_out,
        "pages_read": sum(1 for f in files_out for p in f["pages"] if not p.get("error")),
        "pages_failed": sum(1 for f in files_out for p in f["pages"] if p.get("error")),
        "seconds": round(time.time() - started, 1),
    }
    result_path(thread_id, batch_id).write_text(json.dumps(result, indent=1))
    write_status(thread_id, batch_id, state="done", progress=100, status="Finished reading your files")
    return {k: result[k] for k in ("batch_id", "pages_read", "pages_failed", "seconds")}


@app.task(bind=True, name="helper_quote.read_uploads", max_retries=1, default_retry_delay=20)
def read_uploads_task(self, thread_id: str, batch_id: str) -> Dict[str, Any]:
    def celery_progress(pct: int, status: str) -> None:
        self.update_state(state="PROGRESS", meta={"progress": pct, "status": status})

    try:
        return process_batch(thread_id, batch_id, on_progress=celery_progress)
    except Exception as exc:  # noqa: BLE001
        if self.request.retries < self.max_retries:
            write_status(thread_id, batch_id, state="running", progress=5, status="Retrying...")
            raise self.retry(exc=exc)
        write_status(thread_id, batch_id, state="error", progress=100,
                     status="Couldn't read the files", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        raise
