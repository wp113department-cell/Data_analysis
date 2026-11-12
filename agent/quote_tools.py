"""
quote_tools.py
--------------
The agent's tools. They give the agent *access* (price sheets, memory, a
calculator, the quote document) - all reasoning and decisions (which line,
which SKU, code conversions, handle counts, countertop size) are made by the
agent itself, guided by the owner's playbook.

Price sheet (read-only, live)
    list_price_sheets      brands, tabs, last updated
    sheet_overview         top + bottom rows of every tab in one call (layout learning)
    read_price_sheet       raw cells of a row range
    search_price_sheet     rows containing given texts (SKUs, words)
Math
    calculator             exact arithmetic
Clarify (human-in-the-loop)
    ask_customer           pause the graph, show clickable options, resume with the answer
Quote
    create_quote           numbered quote; refuses prices that aren't in the sheet
    get_saved_quote        reopen a past quote
Long-term memory
    find_customer          returning customer: details, preferences, past quotes
    save_customer_details  remember contact details / preferences right away
    save_sheet_notes       remember a workbook's layout (self-learned, saves tokens)
    learn_rule             (staff only) teach a new business rule
Escalation
    hand_off_to_designer   flag the chat "Needs you" for a human designer
Uploads
    read_uploaded_files    what the vision worker found in the customer's layouts / photos / PDFs
"""

from __future__ import annotations

import ast
import json
import operator
from datetime import date, timedelta
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from agent import memory
from agent.agents import load_settings
from agent.sheets import get_workbooks


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _configurable(config: Optional[RunnableConfig]) -> dict:
    return (config or {}).get("configurable") or {}


def _is_staff(config: Optional[RunnableConfig]) -> bool:
    return _configurable(config).get("role") == "staff"


def _money(x: float) -> float:
    return round(float(x) + 1e-9, 2)


# ---------------------------------------------------------------------------
# Price sheet tools
# ---------------------------------------------------------------------------

@tool
def list_price_sheets() -> str:
    """List every brand price workbook that is loaded: brand key, display name,
    tabs with their sizes, and when the sheet was last updated."""
    return _json(get_workbooks().list_workbooks())


@tool
def sheet_overview(brand: str) -> str:
    """One-call overview of a brand's workbook: the top 30 and last 10 non-empty rows of
    every tab. Use it ONCE to learn a sheet's layout (line options, multiplier, header row,
    fees), then save_sheet_notes."""
    guide = next((v for b, v in memory.sheet_guides().items() if b.lower() == brand.strip().lower()), None)
    if guide:  # a verified guide already describes this layout - no need for a 3k-token sheet dump
        return f"{brand} has a VERIFIED layout guide - use it (also in SHEET NOTES):\n{guide['notes']}"
    try:
        notes_saved = any(b.lower() == brand.strip().lower() for b in memory.sheet_notes())
        return get_workbooks().overview(brand) + ("" if notes_saved else (
            "\n\nNEXT: call save_sheet_notes now with: product tab, header row number, the customer-facing line "
            "names and the price column each maps to (by order), multiplier value, assembly %. Then continue."))
    except Exception as exc:  # noqa: BLE001
        return _json({"error": str(exc)})


@tool
def read_price_sheet(brand: str, tab: str, start_row: int, end_row: int, header_row: Optional[int] = None) -> str:
    """Read raw cells from a brand's price workbook (max 80 rows per call).
    Use it to understand the sheet layout (settings, line options, multiplier,
    header row) or to browse a section of products. Pass header_row to label
    each cell with its column header."""
    try:
        return get_workbooks().read_range(brand, tab, start_row, end_row, header_row)
    except Exception as exc:  # noqa: BLE001
        return _json({"error": str(exc)})


@tool
def search_price_sheet(brand: str, texts: List[str], tab: Optional[str] = None, header_row: Optional[int] = None,
                       columns: Optional[List[str]] = None) -> str:
    """Look up many codes/words in one call, e.g. ['W2136', 'LS36']. Each result says EXACT MATCH,
    or NO EXACT MATCH with the closest real rows (different items - ask, never substitute).
    Pass tab + header_row to label cells, and columns (e.g. ['B','C','L'] = code, description,
    the chosen line's price column) to get short results."""
    try:
        return get_workbooks().search(brand, texts, tab, header_row, columns=columns)
    except Exception as exc:  # noqa: BLE001
        return _json({"error": str(exc)})


@tool
def save_sheet_notes(brand: str, notes: str) -> str:
    """Save a brand workbook's LAYOUT so future chats skip re-reading it: product tab, header
    row, every customer-facing line name with its price column (e.g. 'Colonial White = L'),
    multiplier value, assembly %. Layout only - item prices are always looked up fresh."""
    sheets = {wb["brand"].lower(): wb for wb in get_workbooks().list_workbooks() if "error" not in wb}
    wb = sheets.get(brand.strip().lower())
    if not wb:
        return _json({"saved": False, "reason": f"Unknown brand '{brand}'"})
    if any(b.lower() == wb["brand"].lower() for b in memory.sheet_guides()):
        return _json({"saved": False, "reason": f"{wb['brand']} has a verified guide (see SHEET NOTES) - use it."})
    memory.save_sheet_notes(wb["brand"], notes, wb["last_updated"])
    return _json({"saved": True})


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "round":
        return round(*[_eval(a) for a in node.args])
    raise ValueError("Only numbers, + - * / ** %, parentheses and round() are allowed.")


@tool
def calculator(expression: str) -> str:
    """Evaluate arithmetic exactly, e.g. '12*2 + 14*2' or '412*2 + 15.5'.
    Always use this instead of doing math in your head."""
    try:
        return _json({"expression": expression, "result": _eval(ast.parse(expression, mode="eval"))})
    except Exception as exc:  # noqa: BLE001
        return _json({"error": str(exc)})


# ---------------------------------------------------------------------------
# Quote
# ---------------------------------------------------------------------------

class QuoteItem(BaseModel):
    sku: str = Field(description="SKU as written in the sheet")
    requested_as: Optional[str] = Field(default=None, description="Customer's code if different")
    description: str
    quantity: int = Field(ge=1)
    msrp: float = Field(description="Sheet price for the chosen line (before multiplier)")
    sheet_tab: str
    sheet_row: int
    note: Optional[str] = None


class QuoteExtra(BaseModel):
    label: str = Field(description="e.g. Countertop (estimate), Handles, Installation")
    quantity: float
    unit: str = Field(description="e.g. sq ft, handles, cabinets")
    rate: float = Field(description="Price per unit (playbook)")
    detail: str = Field(description="How quantity was worked out")


class CreateQuoteInput(BaseModel):
    customer_name: str
    customer_phone: str
    customer_email: Optional[str] = None
    customer_address: Optional[str] = None
    brand: str
    line: str = Field(description="Customer-facing line, e.g. Colonial White")
    multiplier: float = Field(description="From the sheet")
    items: List[QuoteItem]
    assembled: bool
    assembly_fee_percent: float = Field(description="From the sheet, e.g. 10")
    shipping: bool
    extras: List[QuoteExtra] = Field(default_factory=list)
    unpriced_items: List[str] = Field(default_factory=list, description="Codes a designer must review")
    revises: Optional[str] = Field(default=None, description="Quote number this one replaces, when editing a quote")
    source_upload_id: Optional[str] = Field(default=None, description="Upload id, when the items came from attached files")
    customer_confirmed_list: bool = Field(default=False, description="True only after the customer said yes to the final list")
    notes: str = ""


@tool(args_schema=CreateQuoteInput)
def create_quote(
    customer_name: str,
    customer_phone: str,
    brand: str,
    line: str,
    multiplier: float,
    items: List[Any],
    assembled: bool,
    assembly_fee_percent: float,
    shipping: bool,
    customer_email: Optional[str] = None,
    customer_address: Optional[str] = None,
    extras: Optional[List[Any]] = None,
    unpriced_items: Optional[List[str]] = None,
    revises: Optional[str] = None,
    source_upload_id: Optional[str] = None,
    customer_confirmed_list: bool = False,
    notes: str = "",
    config: RunnableConfig = None,
) -> str:
    """Create the numbered quote (checks SKUs/MSRPs/multiplier against the sheet,
    totals it, saves it, shows the quote card to the customer)."""
    if not any(b.lower() == brand.strip().lower() for b in memory.sheet_notes()):
        return _json({"status": "not_created",
                      "problems": [f"No sheet notes for {brand} yet. Call save_sheet_notes with the layout you "
                                   "read (product tab, header row, line -> price column map, multiplier, "
                                   "assembly %), then call create_quote again."]})
    as_dict = lambda x: x if isinstance(x, dict) else x.model_dump()  # noqa: E731
    items = [as_dict(i) for i in items]
    extras = [as_dict(x) for x in (extras or [])]
    workbooks = get_workbooks()

    # Grounding check: every SKU and MSRP must really be on the cited sheet row.
    problems = []
    for it in items:
        try:
            row = workbooks.row_values(brand, it["sheet_tab"], int(it["sheet_row"]))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{it['sku']}: cannot read {it['sheet_tab']} row {it['sheet_row']} ({exc})")
            continue
        texts = {str(v).strip().upper() for v in row if v is not None}
        numbers = {float(v) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if it["sku"].strip().upper() not in texts:
            shown = [str(v) for v in row if isinstance(v, str) and v.strip() and not v.startswith("=")][:2]
            problems.append(f"{it['sku']}: not found on {it['sheet_tab']} row {it['sheet_row']} - that row contains "
                            f"{shown}; use the SKU exactly as written there (fix it yourself, no need to ask)")
        if float(it["msrp"]) not in numbers:
            problems.append(f"{it['sku']}: MSRP {it['msrp']} is not a value on {it['sheet_tab']} row {it['sheet_row']}")
    sheet_info = next((wb for wb in workbooks.list_workbooks() if wb.get("brand", "").lower() == brand.lower()), {})
    try:
        if "multiplier" in sheet_info:  # configured cell: must match its LIVE value exactly
            live = float(sheet_info["multiplier"])
            if abs(float(multiplier) - live) > 1e-9:
                problems.append(f"Multiplier must be {live} (live value of {sheet_info['multiplier_cell']}), not {multiplier}")
            raise StopIteration
        tabs = sheet_info.get("tabs", [])
        sheet_numbers = {
            float(v)
            for t in tabs
            for r in range(1, min(t["rows"], 30) + 1)
            for v in workbooks.row_values(brand, t["name"], r)
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if float(multiplier) not in sheet_numbers:
            problems.append(f"Multiplier {multiplier} was not found in the top of the {brand} sheet")
    except StopIteration:
        pass
    except Exception as exc:  # noqa: BLE001
        problems.append(f"Could not verify multiplier: {exc}")
    thread_id_ = _configurable(config).get("thread_id")
    upload = (memory.get_thread(thread_id_) or {}).get("active_upload") if thread_id_ else None
    if upload and not upload.get("quoted"):
        # Quote built from uploaded files: every line must be accounted for, and the customer must confirm.
        lines = upload.get("lines", [])
        covered = len(items) + len(unpriced_items or [])
        if covered < len(lines):
            problems.append(
                f"The customer's files have {len(lines)} lines but this quote accounts for only {covered} "
                f"({len(items)} priced + {len(unpriced_items or [])} unpriced). Every line must be either a priced item "
                "or listed in unpriced_items (e.g. 'B15 trash pull-out - not in HCC, designer to confirm'). "
                "Lines from the files: " + "; ".join(lines))
        if not customer_confirmed_list:
            problems.append("These items came from the customer's files: show the final list (qty x code, plus anything "
                            "unpriced) with ask_customer and get a yes first, then call create_quote with "
                            "customer_confirmed_list=true.")
    if problems:
        return _json({"status": "not_created", "problems": problems,
                      "fix": "Re-read the sheet rows and correct the SKU/MSRP/row/multiplier, then call create_quote again."})

    settings = load_settings()
    company = settings.get("company", {})
    lines = []
    for it in items:
        unit = _money(float(it["msrp"]) * float(multiplier))
        lines.append({
            "sku": it["sku"], "requested_as": it.get("requested_as"), "description": it["description"],
            "quantity": int(it["quantity"]), "unit_price": unit, "total": _money(unit * int(it["quantity"])),
            "msrp": float(it["msrp"]), "note": it.get("note"),
        })
    subtotal = _money(sum(line["total"] for line in lines))
    assembly_fee = _money(subtotal * float(assembly_fee_percent) / 100) if assembled else 0.0
    extra_rows = [{**x, "amount": _money(float(x["quantity"]) * float(x["rate"]))} for x in extras]
    extras_total = _money(sum(x["amount"] for x in extra_rows))
    grand_total = _money(subtotal + assembly_fee + extras_total)

    cust_key = memory.upsert_customer(customer_phone, name=customer_name, email=customer_email, address=customer_address)
    if not cust_key:
        return _json({"status": "not_created", "problems": ["A valid customer phone number is required."]})

    today = date.today()
    quote = {
        "quote_number": memory.next_quote_number(company.get("quote_number_prefix", "HQ-")),
        "created_at": memory.now(),
        "date": today.isoformat(),
        "valid_until": (today + timedelta(days=int(company.get("quote_valid_days", 30)))).isoformat(),
        "company": {k: company.get(k, "") for k in ("name", "phone", "email", "address")},
        "customer": {"name": customer_name, "phone": customer_phone,
                     "email": customer_email or "", "address": customer_address or ""},
        "brand": brand,
        "brand_display": settings.get("price_sheets", {}).get(brand, {}).get("display_name", brand),
        "line": line,
        "multiplier": float(multiplier),
        "items": lines,
        "unpriced_items": unpriced_items or [],
        "assembled": assembled,
        "assembly_fee_percent": float(assembly_fee_percent),
        "shipping": shipping,
        "extras": extra_rows,
        "totals": {"subtotal": subtotal, "assembly_fee": assembly_fee,
                   "extras_total": extras_total, "grand_total": grand_total},
        "notes": notes,
        "revises": revises,
        "footer": company.get("quote_footer", ""),
        "thread_id": _configurable(config).get("thread_id"),
    }
    memory.save_quote(cust_key, quote)
    if upload and not upload.get("quoted") and thread_id_:
        memory.update_thread(thread_id_, active_upload={**upload, "quoted": True})
    if quote["thread_id"]:
        memory.update_thread(quote["thread_id"], customer=cust_key, status=f"Quote {quote['quote_number']}")

    return _json({
        "status": "created",
        "quote_number": quote["quote_number"],
        **({"revises": revises} if revises else {}),
        "items": [{"sku": l["sku"], "qty": l["quantity"], "unit_price": l["unit_price"], "total": l["total"]} for l in lines],
        "subtotal": subtotal, "assembly_fee": assembly_fee,
        "extras": [{"label": x["label"], "amount": x["amount"]} for x in extra_rows],
        "grand_total": grand_total, "valid_until": quote["valid_until"],
        "shown_to_customer": "The quote card with a PDF download is displayed below your reply - don't repeat the table.",
    })


@tool
def get_saved_quote(quote_number: str, phone: str) -> str:
    """Reopen a past quote by its number (e.g. HQ-1003) to review or revise it.
    Requires the phone number the quote was made for."""
    quote = memory.get_quote(quote_number)
    if not quote or memory.customer_key(quote["customer"]["phone"]) != memory.customer_key(phone):
        return _json({"found": False})
    return _json({"found": True, **{k: v for k, v in quote.items() if k not in ("footer", "company")}})


# ---------------------------------------------------------------------------
# Long-term memory tools
# ---------------------------------------------------------------------------

@tool
def find_customer(phone: Optional[str] = None, name: Optional[str] = None, config: RunnableConfig = None) -> str:
    """Look up a returning customer: saved contact details, everything we remember
    about them (preferences, kitchen details) and their past quotes.
    Customers are looked up by phone; staff may also search by name."""
    if phone:
        key = memory.customer_key(phone)
        profile = memory.get_customer(key) if key else None
        if not profile:
            return _json({"found": False})
        thread_id = _configurable(config).get("thread_id")
        if thread_id:
            memory.update_thread(thread_id, customer=key)
        past = [{"quote_number": q["quote_number"], "date": q["date"], "line": q["line"],
                 "items": [f"{i['quantity']} x {i['sku']}" for i in q["items"]],
                 "grand_total": q["totals"]["grand_total"]} for q in memory.quotes_for_customer(key)]
        return _json({"found": True, "profile": profile, "past_quotes": past})
    if name:
        if not _is_staff(config):
            return _json({"found": False, "message": "For privacy, ask the customer for their phone number."})
        return _json({"matches": memory.find_customers_by_name(name)})
    return _json({"error": "Give a phone number (or a name, staff only)."})


@tool
def save_customer_details(
    phone: str,
    name: Optional[str] = None,
    email: Optional[str] = None,
    address: Optional[str] = None,
    facts: Optional[List[str]] = None,
    config: RunnableConfig = None,
) -> str:
    """Remember a customer right away: contact details plus short facts worth
    keeping for future chats (e.g. 'prefers white shaker', 'L-shaped kitchen 12ft x 14ft',
    'wants installation', 'remodel finishing in March')."""
    key = memory.upsert_customer(phone, name=name, email=email, address=address, facts=facts)
    if not key:
        return _json({"saved": False, "reason": "A valid phone number is required."})
    thread_id = _configurable(config).get("thread_id")
    if thread_id:
        memory.update_thread(thread_id, customer=key)
    return _json({"saved": True})


@tool
def learn_rule(rule: str, config: RunnableConfig = None) -> str:
    """STAFF ONLY: permanently learn a business rule the owner/staff teaches,
    e.g. 'Espresso color goes to JSI' or 'BCB36 from layouts = HCC BBC42/45'.
    It is applied in every future conversation."""
    if not _is_staff(config):
        return _json({"saved": False, "reason": "Only staff can teach business rules."})
    memory.add_learned_rule(rule)
    return _json({"saved": True, "rule": rule})


# ---------------------------------------------------------------------------
# Human-in-the-loop: clarify with clickable options
# ---------------------------------------------------------------------------

@tool
def ask_customer(question: str, options: List[str], allow_multiple: bool = False) -> str:
    """Pause and ask the customer to choose when their info is wrong, missing or
    ambiguous (unknown code, color we don't carry, unclear quantity, assembled or
    not, ...). Give 2-6 short, concrete options taken from the sheet/playbook; the
    customer clicks one (or several if allow_multiple) or types their own answer.
    Call it ALONE (not together with other tools). Returns the customer's answer."""
    answer = interrupt({"question": question, "options": options[:6], "allow_multiple": allow_multiple})
    return _json({"customer_answer": answer})


@tool
def hand_off_to_designer(reason: str, summary: str, urgency: str = "normal", config: RunnableConfig = None) -> str:
    """Flag this chat for a human designer when you can't resolve it well: the customer
    asks for a person, stays frustrated after you tried, needs something outside the
    price sheets (custom sizes, design advice, a brand not loaded), or reports a
    problem with an order. `summary` = what they need + what's done so far.
    urgency: 'normal' or 'urgent'. Then tell the customer a designer will follow up."""
    thread_id = _configurable(config).get("thread_id")
    if not thread_id:
        return _json({"saved": False})
    memory.add_handoff(thread_id, reason, summary, urgency if urgency in ("normal", "urgent") else "normal")
    return _json({"saved": True, "status": "A designer has been notified and will follow up."})


@tool
def read_uploaded_files(upload_id: str, config: RunnableConfig = None) -> str:
    """Get what was found in the files (layout drawings, photos of lists, PDFs) the customer
    attached - codes exactly as written, quantities, locations, colors, measurements,
    customer details and anything unsure. Treat it as a draft to check, not as final."""
    from agent.uploads import jobs

    thread_id = _configurable(config).get("thread_id")
    if not thread_id:
        return _json({"error": "No conversation id."})
    st = jobs.status(thread_id, upload_id)
    if st.get("state") == "done":
        text = jobs.result_text(thread_id, upload_id)
        if not text:
            return _json({"error": "Result missing."})
        lines = jobs.extracted_lines(thread_id, upload_id)
        memory.update_thread(thread_id, active_upload={"id": upload_id, "lines": lines, "quoted": False})
        return text + (
            "\n\nNEXT (keep it to a few steps): 1) convert designer codes per the playbook (BER36-L->LS36, drop a "
            "trailing L/R hinge suffix - no question needed); 2) ONE search_price_sheet with ALL codes, using the "
            "guide's tab, header_row and columns=[code col, description col, the line's price col]; 3) for each code "
            "with NO EXACT MATCH or UNSURE, ask_customer with the real alternatives; 4) ONE ask_customer showing the "
            "final list (qty x code) for a yes; 5) create_quote with source_upload_id=" + upload_id +
            " and customer_confirmed_list=true. Use the multiplier from PRICE SHEETS LOADED.")
    if st.get("state") == "error":
        return _json({"state": "error", "detail": st.get("error"),
                      "next": "Apologise, and ask the customer to type the list or try a clearer photo/PDF."})
    return _json({"state": st.get("state"), "progress": st.get("progress"), "status": st.get("status"),
                  "next": "Still reading - tell the customer you're reading their files and will continue "
                          "automatically when done. Don't guess the contents."})


CUSTOMER_TOOLS = [
    sheet_overview, search_price_sheet, save_sheet_notes, calculator,
    ask_customer, create_quote, get_saved_quote, find_customer, hand_off_to_designer, read_uploaded_files,
]
STAFF_TOOLS = CUSTOMER_TOOLS + [learn_rule]

TOOL_LABELS = {
    "list_price_sheets": "Checking price lists",
    "sheet_overview": "Learning the price sheet layout",
    "read_price_sheet": "Reading the price sheet",
    "search_price_sheet": "Looking up items in the price sheet",
    "save_sheet_notes": "Learning the price sheet layout",
    "calculator": "Calculating",
    "ask_customer": "Asking you to choose",
    "create_quote": "Creating the quote",
    "get_saved_quote": "Opening a saved quote",
    "find_customer": "Checking customer history",
    "save_customer_details": "Saving customer details",
    "learn_rule": "Learning a new rule",
    "hand_off_to_designer": "Handing over to a designer",
    "read_uploaded_files": "Reading your files",
}
