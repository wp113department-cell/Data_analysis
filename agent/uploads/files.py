"""
files.py
--------
Validate, store and prepare customer uploads for the vision model.

    validate_batch()  -> checks counts (images / PDFs), size, and the REAL file type
                         (by content, not by the file name)
    save_batch()      -> writes the files under storage/uploads/<thread>/<batch>/
    prepare_pages()   -> every PDF page / photo becomes a clean PNG for the vision
                         model; PDFs also keep their text layer (exact text for free)

PDF -> image conversion adapted from zerox (getomni-ai/zerox, MIT); pypdfium2 is
used instead of pdf2image so no system packages (poppler) are needed.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps

from agent.agents import STORAGE_DIR, load_settings

try:  # iPhone photos (HEIC/HEIF)
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # noqa: BLE001 - HEIC support is optional
    pass

UPLOAD_DIR = STORAGE_DIR / "uploads"


def upload_settings() -> Dict[str, Any]:
    defaults = {"max_images": 3, "max_pdfs": 2, "max_file_mb": 10, "max_pdf_pages": 10,
                "image_types": ["png", "jpg", "jpeg", "webp", "heic", "heif"],
                "render_dpi": 170, "max_image_side_px": 2400, "parallel_pages": 4,
                "redis_url": "redis://127.0.0.1:6379/0", "task_time_limit_seconds": 300}
    return {**defaults, **{k: v for k, v in load_settings().get("uploads", {}).items() if not k.startswith("_")}}


@dataclass
class StoredFile:
    name: str           # original file name (display only)
    kind: str           # "pdf" | "image"
    path: str           # where it is stored
    sha256: str
    size_bytes: int
    pages: int = 1


@dataclass
class Batch:
    batch_id: str
    thread_id: str
    created_at: str
    files: List[StoredFile] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "files": [asdict(f) for f in self.files]}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def sniff_kind(data: bytes) -> Optional[str]:
    """Real file type from the content: 'pdf', 'image' or None."""
    if data[:5] == b"%PDF-":
        return "pdf"
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        return "image"
    except Exception:  # noqa: BLE001
        return None


def pdf_page_count(data: bytes) -> int:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    try:
        return len(pdf)
    finally:
        pdf.close()


def validate_batch(files: List[Tuple[str, bytes]]) -> Tuple[List[Tuple[str, bytes, str]], List[str]]:
    """
    Check a message's attachments. Returns (accepted [(name, data, kind)], problems [str]).
    Limits come from settings.json -> uploads (per message).
    """
    cfg = upload_settings()
    max_bytes = int(cfg["max_file_mb"]) * 1024 * 1024
    accepted, problems = [], []
    images = pdfs = 0
    for name, data in files:
        kind = sniff_kind(data)
        if kind is None:
            problems.append(f"{name}: not a PDF or a supported image (PNG, JPG, WEBP, HEIC).")
            continue
        if len(data) > max_bytes:
            problems.append(f"{name}: {len(data) / 1048576:.1f} MB - the limit is {cfg['max_file_mb']} MB per file.")
            continue
        if kind == "pdf":
            if pdfs >= int(cfg["max_pdfs"]):
                problems.append(f"{name}: only {cfg['max_pdfs']} PDFs per message.")
                continue
            try:
                pages = pdf_page_count(data)
            except Exception:  # noqa: BLE001
                problems.append(f"{name}: this PDF can't be opened (damaged or password-protected).")
                continue
            if pages > int(cfg["max_pdf_pages"]):
                problems.append(f"{name}: {pages} pages - the limit is {cfg['max_pdf_pages']} pages per PDF.")
                continue
            pdfs += 1
        else:
            if images >= int(cfg["max_images"]):
                problems.append(f"{name}: only {cfg['max_images']} images per message.")
                continue
            images += 1
        accepted.append((name, data, kind))
    return accepted, problems


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _safe_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)[:80].strip("._") or "file"
    return stem


def save_batch(thread_id: str, accepted: List[Tuple[str, bytes, str]]) -> Batch:
    batch = Batch(batch_id=uuid.uuid4().hex[:12], thread_id=thread_id,
                  created_at=datetime.now().isoformat(timespec="seconds"))
    folder = UPLOAD_DIR / _safe_name(thread_id) / batch.batch_id
    folder.mkdir(parents=True, exist_ok=True)
    for i, (name, data, kind) in enumerate(accepted, 1):
        path = folder / f"{i:02d}_{_safe_name(name)}"
        path.write_bytes(data)
        batch.files.append(StoredFile(
            name=Path(name).name, kind=kind, path=str(path), sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data), pages=pdf_page_count(data) if kind == "pdf" else 1))
    (folder / "batch.json").write_text(json.dumps(batch.to_dict(), indent=1))
    return batch


def load_batch(thread_id: str, batch_id: str) -> Optional[Batch]:
    meta = UPLOAD_DIR / _safe_name(thread_id) / batch_id / "batch.json"
    if not meta.exists():
        return None
    d = json.loads(meta.read_text())
    return Batch(batch_id=d["batch_id"], thread_id=d["thread_id"], created_at=d["created_at"],
                 files=[StoredFile(**f) for f in d["files"]])


def result_path(thread_id: str, batch_id: str) -> Path:
    return UPLOAD_DIR / _safe_name(thread_id) / batch_id / "result.json"


# ---------------------------------------------------------------------------
# Pages for the vision model
# ---------------------------------------------------------------------------

@dataclass
class Page:
    file_name: str
    page_no: int          # 1-based
    png: bytes            # cleaned image for the vision model
    text_layer: str = ""  # exact text from a digital PDF page (empty for photos/scans)


def _clean_image(img: Image.Image, max_side: int) -> bytes:
    """Upright (EXIF), RGB, no metadata (drops GPS), downscaled to max_side, PNG."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def prepare_pages(stored: StoredFile) -> List[Page]:
    cfg = upload_settings()
    max_side = int(cfg["max_image_side_px"])
    data = Path(stored.path).read_bytes()
    if stored.kind == "image":
        with Image.open(io.BytesIO(data)) as img:
            return [Page(stored.name, 1, _clean_image(img, max_side))]

    import pypdfium2 as pdfium
    from pypdf import PdfReader

    try:
        texts = [(p.extract_text() or "").strip() for p in PdfReader(io.BytesIO(data)).pages]
    except Exception:  # noqa: BLE001 - scanned/odd PDFs: rely on the images
        texts = []
    pdf = pdfium.PdfDocument(data)
    pages: List[Page] = []
    try:
        scale = int(cfg["render_dpi"]) / 72
        for i in range(min(len(pdf), int(cfg["max_pdf_pages"]))):
            bitmap = pdf[i].render(scale=scale)
            img = bitmap.to_pil()
            pages.append(Page(stored.name, i + 1, _clean_image(img, max_side),
                              texts[i] if i < len(texts) else ""))
    finally:
        pdf.close()
    return pages
