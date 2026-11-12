"""
jobs.py
-------
Submit upload-reading jobs and check on them.

    submit()       -> Celery worker if one is running, otherwise a background thread
                      in the app (so the app still works without Redis/workers)
    status()       -> queued / running (progress %) / done / error
    result_text()  -> compact text the agent reads (what's on every page)
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional

from agent.uploads.files import Batch, load_batch, result_path

_worker_cache: Dict[str, Any] = {"at": 0.0, "alive": False}


def worker_alive() -> bool:
    """Is at least one Celery worker online? (cached 20 s)"""
    if time.time() - _worker_cache["at"] < 20:
        return _worker_cache["alive"]
    alive = False
    try:
        import redis

        from agent.uploads.celery_app import REDIS_URL, app

        # Fail fast when there is no Redis at all (e.g. Streamlit Cloud): don't let Celery retry connecting.
        redis.Redis.from_url(REDIS_URL, socket_connect_timeout=0.5, socket_timeout=0.5).ping()
        alive = bool(app.control.ping(timeout=1.0))
    except Exception:  # noqa: BLE001 - no broker / no worker -> read files in a background thread
        alive = False
    _worker_cache.update(at=time.time(), alive=alive)
    return alive


def submit(batch: Batch) -> str:
    """Start reading a batch in the background. Returns 'celery' or 'thread'."""
    from agent.uploads.tasks import process_batch, write_status

    write_status(batch.thread_id, batch.batch_id, state="queued", progress=0, status="Waiting to start")
    if worker_alive():
        from agent.uploads.tasks import read_uploads_task

        read_uploads_task.apply_async(args=[batch.thread_id, batch.batch_id], task_id=batch.batch_id)
        return "celery"

    def run() -> None:
        try:
            process_batch(batch.thread_id, batch.batch_id)
        except Exception as exc:  # noqa: BLE001
            write_status(batch.thread_id, batch.batch_id, state="error", progress=100,
                         status="Couldn't read the files", error=f"{type(exc).__name__}: {str(exc)[:300]}")

    threading.Thread(target=run, daemon=True, name=f"upload-{batch.batch_id}").start()
    return "thread"


def status(thread_id: str, batch_id: str) -> Dict[str, Any]:
    path = result_path(thread_id, batch_id).with_name("status.json")
    if not path.exists():
        return {"state": "unknown", "progress": 0, "status": "No such upload"}
    return json.loads(path.read_text())


def result_text(thread_id: str, batch_id: str) -> Optional[str]:
    """What the vision model found, as compact text for the agent (None if not ready)."""
    path = result_path(thread_id, batch_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    lines: List[str] = [f"Upload {batch_id}: {data['pages_read']} page(s) read"
                        + (f", {data['pages_failed']} failed" if data["pages_failed"] else "")]
    for f in data["files"]:
        for p in f["pages"]:
            head = f"\n== {f['name']} page {p['page']}"
            if p.get("error"):
                lines.append(f"{head}: COULD NOT READ ({p['error']})")
                continue
            if not p.get("readable", True):
                lines.append(f"{head}: unreadable / not cabinet-related. {p.get('notes', '')}")
                continue
            lines.append(f"{head} ({p.get('document_type')})")
            for it in p.get("items", []):
                extra = []
                if it.get("description"):
                    extra.append(it["description"])
                if it.get("location"):
                    extra.append(f"at {it['location']}")
                if it.get("confidence") != "high":
                    alt = f" or {'/'.join(it['alternatives'])}" if it.get("alternatives") else ""
                    extra.append(f"UNSURE ({it.get('confidence')}){alt}")
                lines.append(f"- {it['quantity']} x {it['code']}" + (f" ({'; '.join(extra)})" if extra else ""))
            for label, key in (("Color/style", "color_or_style"), ("Customer details", "customer_details")):
                if p.get(key):
                    lines.append(f"{label}: {p[key]}")
            if p.get("room_measurements"):
                lines.append("Measurements: " + "; ".join(p["room_measurements"]))
            if p.get("notes"):
                lines.append(f"Notes: {p['notes']}")
    return "\n".join(lines)


def batch_files(thread_id: str, batch_id: str) -> List[str]:
    batch = load_batch(thread_id, batch_id)
    return [f.name for f in batch.files] if batch else []


def extracted_lines(thread_id: str, batch_id: str) -> List[str]:
    """Every distinct line the vision reader found ("qty x code (description)"), across all pages."""
    path = result_path(thread_id, batch_id)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    seen, lines = set(), []
    for f in data["files"]:
        for p in f["pages"]:
            for it in p.get("items", []):
                key = (str(it.get("code", "")).strip().upper(), str(it.get("description") or "").strip().lower())
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"{it.get('quantity', 1)} x {it.get('code')}" + (f" ({it['description']})" if it.get("description") else ""))
    return lines
