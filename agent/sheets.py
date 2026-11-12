"""
sheets.py
---------
Generic, read-only spreadsheet access for the agent - the same kind of
operations an Excel/Google-Sheets MCP server exposes:

    list_workbooks()            which brand sheets exist, their tabs and sizes
    read_range(...)             raw cells of a row range
    search(...)                 rows containing a text

There is NO knowledge of the quote layout here. The agent reads the sheet
itself (guided by the owner's playbook) and decides what the cells mean.

Live data: each workbook is re-read when its source changes (file modified
time for Excel, a refresh interval for Google Sheets). To move to an MCP
server later, replace SheetSource.load() - the tools don't change.
"""

from __future__ import annotations

import io
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import openpyxl
from openpyxl.utils import get_column_letter

from agent.agents import load_settings, resolve_path


class SheetSource:
    def __init__(self, brand: str, spec: Dict[str, Any]):
        self.brand = brand
        self.spec = spec

    def version(self) -> Any:
        if self.spec.get("type", "excel") == "google_sheets":
            refresh = max(int(self.spec.get("refresh_seconds", 300)), 30)
            return int(time.time() // refresh)
        return self._excel_file()[0].stat().st_mtime

    def _excel_file(self):
        """(path, encrypted?) - the plain workbook if present, else its encrypted copy (public-repo deploys)."""
        plain = resolve_path(self.spec["path"])
        if plain.exists() or not self.spec.get("encrypted_path"):
            return plain, False
        return resolve_path(self.spec["encrypted_path"]), True

    def load(self):
        """Return (values_workbook, formulas_workbook, label, modified)."""
        if self.spec.get("type", "excel") == "google_sheets":
            url = f"https://docs.google.com/spreadsheets/d/{self.spec['sheet_id']}/export?format=xlsx"
            with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - fixed Google URL
                data = resp.read()
            values = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
            formulas = openpyxl.load_workbook(io.BytesIO(data), data_only=False)
            return values, formulas, "Google Sheet", datetime.now()
        path, encrypted = self._excel_file()
        if encrypted:
            import os

            from cryptography.fernet import Fernet

            key = os.getenv("PRICE_SHEET_KEY")
            if not key:
                raise ValueError("PRICE_SHEET_KEY is missing - needed to open the encrypted price sheet")
            data = Fernet(key.encode()).decrypt(path.read_bytes())  # decrypted in memory only
            values = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
            formulas = openpyxl.load_workbook(io.BytesIO(data), data_only=False)
            return values, formulas, Path(self.spec["path"]).name, datetime.fromtimestamp(path.stat().st_mtime)
        values = openpyxl.load_workbook(path, data_only=True)
        formulas = openpyxl.load_workbook(path, data_only=False)
        return values, formulas, path.name, datetime.fromtimestamp(path.stat().st_mtime)


class Workbooks:
    """Caches every configured workbook and reloads stale ones on access."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _get(self, brand: str) -> Dict[str, Any]:
        sheets_cfg = load_settings().get("price_sheets", {})
        match = next((b for b in sheets_cfg if b.lower() == brand.strip().lower()), None)
        if match is None:
            raise KeyError(f"No price sheet configured for '{brand}'. Configured: {list(sheets_cfg)}")
        spec = sheets_cfg[match]
        source = SheetSource(match, spec)
        with self._lock:
            entry = self._cache.get(match)
            version = source.version()
            if entry is None or entry["version"] != version or entry["spec"] != spec:
                values, formulas, label, modified = source.load()
                entry = {
                    "version": version, "spec": spec, "values": values, "formulas": formulas,
                    "label": label, "modified": modified, "display_name": spec.get("display_name", match),
                }
                self._cache[match] = entry
            return entry

    # -- operations exposed to the agent -----------------------------------

    def list_workbooks(self) -> List[Dict[str, Any]]:
        out = []
        for brand in load_settings().get("price_sheets", {}):
            try:
                e = self._get(brand)
                out.append({
                    "brand": brand,
                    "display_name": e["display_name"],
                    "source": e["label"],
                    "last_updated": e["modified"].isoformat(timespec="minutes"),
                    "tabs": [{"name": ws.title, "rows": ws.max_row, "columns": ws.max_column}
                             for ws in e["values"].worksheets],
                    **({"multiplier": self.cell(brand, e["spec"]["multiplier_cell"]),
                        "multiplier_cell": e["spec"]["multiplier_cell"]}
                       if e["spec"].get("multiplier_cell") else {}),
                })
            except Exception as exc:  # noqa: BLE001 - reported to the agent/UI
                out.append({"brand": brand, "error": str(exc)})
        return out

    def cell(self, brand: str, ref: str) -> Any:
        """Live value of one cell, e.g. 'Worksheet!H7' (re-read whenever the sheet changes)."""
        e = self._get(brand)
        tab, _, addr = ref.rpartition("!")
        ws = e["values"][tab or e["values"].sheetnames[0]][addr]
        return self._cell(e, ws.parent.title, ws.row, ws.column)

    def _cell(self, e, ws_name: str, row: int, col: int) -> Any:
        value = e["values"][ws_name].cell(row, col).value
        if value is None:
            raw = e["formulas"][ws_name].cell(row, col).value
            if isinstance(raw, str) and raw.startswith("="):
                return raw  # uncached formula: let the agent follow the reference
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, datetime):
            return value.date().isoformat()
        return value

    def _row(self, e, ws_name: str, row: int, max_col: int, header_row: Optional[int],
             columns: Optional[List[str]] = None) -> str:
        """One row as a compact line: 'B Item #=W2136 | C Description=... | L Mercury White=412'."""
        cells = []
        for col in range(1, max_col + 1):
            v = self._cell(e, ws_name, row, col)
            if v is None or v == "":
                continue
            key = get_column_letter(col)
            if columns and key not in columns:
                continue
            if header_row:
                label = e["values"][ws_name].cell(header_row, col).value
                if label not in (None, ""):
                    key = f"{key} {label}"
            cells.append(f"{key}={v}")
        return " | ".join(cells)

    def read_range(self, brand: str, tab: str, start_row: int, end_row: int,
                   header_row: Optional[int] = None) -> str:
        e = self._get(brand)
        ws = e["values"][tab]
        end_row = min(end_row, ws.max_row, start_row + 79)
        lines = [f"[{brand} / {tab}] rows {start_row}-{end_row} (empty cells omitted, max 80 rows per call)"]
        for r in range(max(start_row, 1), end_row + 1):
            text = self._row(e, tab, r, ws.max_column, header_row)
            if text:
                lines.append(f"row {r}: {text}")
        return "\n".join(lines)

    def search(self, brand: str, texts: List[str], tab: Optional[str] = None,
               header_row: Optional[int] = None, limit_per_text: int = 12,
               columns: Optional[List[str]] = None) -> str:
        e = self._get(brand)
        tabs = [tab] if tab else [ws.title for ws in e["values"].worksheets]
        lines = [f"[{brand}] search results"]

        def find(needle: str, limit: int, exact: bool) -> List[str]:
            """exact=True: a cell equals the text; exact=False: a cell contains it."""
            hits: List[str] = []
            for name in tabs:
                ws = e["values"][name]
                for row in ws.iter_rows():
                    cells = [str(c.value).strip().lower() for c in row if c.value is not None]
                    if any((v == needle) if exact else (needle in v) for v in cells):
                        r = row[0].row
                        hits.append(f"  {name} row {r}: {self._row(e, name, r, ws.max_column, header_row, cols)}")
                        if len(hits) >= limit:
                            return hits
            return hits

        cols = [c.strip().upper() for c in columns] if columns else None
        for text in texts:
            needle = str(text).strip().lower()
            exact = find(needle, 3, exact=True)
            if exact:
                lines.append(f'"{text}": EXACT MATCH')
                lines.extend(exact)
                continue
            partial = find(needle, 6, exact=False)
            if partial:
                lines.append(f'"{text}": NO EXACT MATCH - these rows only CONTAIN "{text}" and are DIFFERENT items '
                             "(ask the customer, never substitute silently):")
                lines.extend(partial)
                continue
            # Nothing contains it: shorten the text step by step and return the closest REAL rows,
            # so alternatives offered to the customer always come from the sheet.
            closest, prefix = [], needle
            while not closest and len(prefix) > 3:
                prefix = prefix[:-1]
                closest = find(prefix, 8, exact=False)
            if closest:
                lines.append(f'"{text}": NO EXACT MATCH. Closest real items (rows containing "{prefix}"):')
                lines.extend(closest)
            else:
                lines.append(f'"{text}": no matches')
        return "\n".join(lines)

    def overview(self, brand: str, top_rows: int = 30, bottom_rows: int = 10) -> str:
        """Top and bottom of every tab in one call (settings/headers usually live at the top,
        totals/fees at the bottom). Generic: no knowledge of any particular layout."""
        e = self._get(brand)
        out = [f"[{brand}] overview of every tab (top {top_rows} and last {bottom_rows} rows that contain text)"]
        for ws in e["values"].worksheets:
            # Layout lives in labelled rows; rows holding only numbers/formulas are data or filler.
            non_empty = [r for r in range(1, ws.max_row + 1)
                         if any(isinstance(c.value, str) and c.value.strip() and not c.value.startswith("=")
                                for c in ws[r])]
            show = non_empty[:top_rows]
            tail = [r for r in non_empty[-bottom_rows:] if r not in show]
            out.append(f"\n== tab '{ws.title}' ({len(non_empty)} rows with text) ==")
            for r in show:
                out.append(f"row {r}: {self._row(e, ws.title, r, ws.max_column, None)}")
            if tail:
                out.append("...")
                for r in tail:
                    out.append(f"row {r}: {self._row(e, ws.title, r, ws.max_column, None)}")
        return "\n".join(out)

    def row_values(self, brand: str, tab: str, row: int) -> List[Any]:
        e = self._get(brand)
        ws = e["values"][tab]
        return [self._cell(e, tab, row, c) for c in range(1, ws.max_column + 1)]


_workbooks = Workbooks()


def get_workbooks() -> Workbooks:
    return _workbooks
