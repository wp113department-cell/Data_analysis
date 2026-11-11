"""
quote_document.py
-----------------
Turns a saved quote into a downloadable PDF (and an HTML version).
Presentation only - the numbers come from create_quote.
"""

from __future__ import annotations

import html
import unicodedata
from typing import Any, Dict

from fpdf import FPDF
from fpdf.fonts import FontFace

INDIGO = (79, 70, 229)
SLATE = (100, 116, 139)
INK = (15, 23, 42)


def _latin1(text: Any) -> str:
    """Core PDF fonts are Latin-1: normalize fancy characters (e.g. full-width brackets)."""
    text = unicodedata.normalize("NFKC", str(text or ""))
    for a, b in (("‑", "-"), ("–", "-"), ("—", "-"), ("‘", "'"), ("’", "'"),
                 ("“", '"'), ("”", '"'), ("×", "x")):
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


def render_quote_pdf(q: Dict[str, Any]) -> bytes:
    t, cust, comp = q["totals"], q["customer"], q["company"]
    pdf = FPDF(format="Letter")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    # Header
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*INK)
    pdf.cell(0, 10, _latin1(comp.get("name") or "Quote"), new_x="LMARGIN", new_y="NEXT")
    contact = "  |  ".join(x for x in (comp.get("address"), comp.get("phone"), comp.get("email")) if x)
    if contact:
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*SLATE)
        pdf.cell(0, 5, _latin1(contact), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*INDIGO)
    pdf.set_line_width(0.6)
    pdf.line(15, pdf.get_y() + 2, 200, pdf.get_y() + 2)
    pdf.ln(6)

    # Customer + quote meta
    top = pdf.get_y()
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*INK)
    pdf.cell(90, 6, "Prepared for", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for line in (cust["name"], cust["phone"], cust.get("email"), cust.get("address")):
        if line:
            pdf.cell(90, 5, _latin1(line), new_x="LMARGIN", new_y="NEXT")
    left_bottom = pdf.get_y()

    meta = [
        ("Quote #", q["quote_number"]), ("Date", q["date"]), ("Valid until", q["valid_until"]),
        ("Brand", q["brand_display"]), ("Color / line", q["line"]),
        ("Assembly", "Assembled" if q["assembled"] else "Not assembled (RTA)"),
    ]
    pdf.set_y(top)
    for label, value in meta:
        pdf.set_x(115)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(28, 5, label)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5, _latin1(value), new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(max(left_bottom, pdf.get_y()) + 6)

    # Items
    pdf.set_font("Helvetica", "", 9)
    head = FontFace(emphasis="BOLD", fill_color=(238, 242, 255), color=INK)
    with pdf.table(col_widths=(24, 92, 14, 25, 25), text_align=("LEFT", "LEFT", "RIGHT", "RIGHT", "RIGHT"),
                   headings_style=head, line_height=6, borders_layout="HORIZONTAL_LINES") as table:
        table.row(["Item #", "Description", "Qty", "Unit price", "Total"])
        for i in q["items"]:
            desc = i["description"] + (f"  ({i['note']})" if i.get("note") else "")
            table.row([_latin1(i["sku"]), _latin1(desc), str(i["quantity"]),
                       f"${i['unit_price']:,.2f}", f"${i['total']:,.2f}"])
    pdf.ln(4)

    # Totals
    rows = [("Cabinets & accessories", t["subtotal"])]
    if q["assembled"]:
        rows.append((f"Assembly ({q['assembly_fee_percent']:g}%)", t["assembly_fee"]))
    for x in q["extras"]:
        rows.append((f"{x['label']} - {x['detail']}", x["amount"]))
    pdf.set_font("Helvetica", "", 10)
    for label, amount in rows:
        pdf.set_x(15)
        pdf.multi_cell(150, 6, _latin1(label), align="R", new_x="RIGHT", new_y="TOP", max_line_height=6)
        pdf.cell(0, 6, f"${amount:,.2f}", align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*INK)
    pdf.line(120, pdf.get_y() + 1, 200, pdf.get_y() + 1)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(150, 8, "Total", align="R")
    pdf.cell(0, 8, f"${t['grand_total']:,.2f}", align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Notes
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*SLATE)
    notes = []
    if q.get("unpriced_items"):
        notes.append("To be confirmed by our designer (not priced): " + ", ".join(q["unpriced_items"]))
    notes.append("Shipping requested - the office will send a separate shipping quote."
                 if q["shipping"] else "Pickup / no shipping.")
    if q.get("notes"):
        notes.append("Notes: " + q["notes"])
    if q.get("footer"):
        notes.append(q["footer"])
    for n in notes:
        pdf.multi_cell(0, 5, _latin1(n), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    return bytes(pdf.output())


def render_quote_html(q: Dict[str, Any]) -> str:
    e = lambda x: html.escape(str(x or ""))  # noqa: E731
    t = q["totals"]
    cust, comp = q["customer"], q["company"]

    rows = "".join(
        f"<tr><td>{e(i['sku'])}</td><td>{e(i['description'])}"
        + (f"<div class='note'>{e(i['note'])}</div>" if i.get("note") else "")
        + f"</td><td class='n'>{i['quantity']}</td><td class='n'>${i['unit_price']:,.2f}</td>"
          f"<td class='n'>${i['total']:,.2f}</td></tr>"
        for i in q["items"]
    )
    summary = [("Cabinets &amp; accessories", t["subtotal"])]
    if q["assembled"]:
        summary.append((f"Assembly ({q['assembly_fee_percent']:g}%)", t["assembly_fee"]))
    for x in q["extras"]:
        summary.append((f"{e(x['label'])}<div class='note'>{e(x['detail'])}</div>", x["amount"]))
    summary_rows = "".join(f"<tr><td>{label}</td><td class='n'>${amt:,.2f}</td></tr>" for label, amt in summary)

    unpriced = ""
    if q.get("unpriced_items"):
        lis = "".join(f"<li>{e(u)}</li>" for u in q["unpriced_items"])
        unpriced = f"<div class='warn'><b>To be confirmed by our designer (not priced):</b><ul>{lis}</ul></div>"

    contact = " &middot; ".join(e(x) for x in (comp.get("address"), comp.get("phone"), comp.get("email")) if x)
    shipping = ("Shipping requested - the office will send a separate shipping quote."
                if q["shipping"] else "Pickup / no shipping.")
    cust_lines = "<br>".join(e(x) for x in (cust["name"], cust["phone"], cust.get("email"), cust.get("address")) if x)

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Quote {e(q['quote_number'])}</title>
<style>
body {{ font-family: Inter, Arial, sans-serif; color:#0F172A; max-width:820px; margin:32px auto; padding:0 16px; }}
h1 {{ margin:0; font-size:26px; }} .accent {{ color:#4F46E5; font-weight:600; }}
.head {{ display:flex; justify-content:space-between; gap:24px; border-bottom:2px solid #4F46E5; padding-bottom:16px; flex-wrap:wrap; }}
.meta td {{ padding:2px 8px; font-size:14px; }} .muted {{ color:#64748B; font-size:13px; }}
table.items, table.sum {{ width:100%; border-collapse:collapse; margin-top:18px; font-size:14px; }}
table.items th {{ text-align:left; background:#EEF2FF; padding:8px; }}
table.items td {{ padding:8px; border-bottom:1px solid #E2E8F0; vertical-align:top; }}
.n {{ text-align:right; white-space:nowrap; }} .note {{ color:#64748B; font-size:12px; }}
table.sum {{ max-width:380px; margin-left:auto; }} table.sum td {{ padding:6px 8px; }}
.total td {{ font-size:18px; font-weight:700; border-top:2px solid #0F172A; }}
.warn {{ background:#FFFBEB; border:1px solid #F59E0B; border-radius:8px; padding:8px 14px; margin-top:16px; font-size:14px; }}
</style></head><body>
<div class="head">
  <div><h1>{e(comp.get('name') or 'Quote')}</h1><div class="muted">{contact}</div>
       <p><b>Prepared for</b><br>{cust_lines}</p></div>
  <table class="meta">
    <tr><td><b>Quote #</b></td><td>{e(q['quote_number'])}</td></tr>
    <tr><td><b>Date</b></td><td>{e(q['date'])}</td></tr>
    <tr><td><b>Valid until</b></td><td>{e(q['valid_until'])}</td></tr>
    <tr><td><b>Brand</b></td><td>{e(q['brand_display'])}</td></tr>
    <tr><td><b>Color / line</b></td><td class="accent">{e(q['line'])}</td></tr>
    <tr><td><b>Assembly</b></td><td>{'Assembled' if q['assembled'] else 'Not assembled (RTA)'}</td></tr>
  </table>
</div>
<table class="items"><tr><th>Item #</th><th>Description</th><th class="n">Qty</th><th class="n">Unit price</th><th class="n">Total</th></tr>{rows}</table>
{unpriced}
<table class="sum">{summary_rows}<tr class="total"><td>Total</td><td class="n">${t['grand_total']:,.2f}</td></tr></table>
<p class="muted">{shipping}</p>
{f"<p><b>Notes:</b> {e(q['notes'])}</p>" if q.get('notes') else ''}
<p class="muted">{e(q.get('footer'))}</p>
</body></html>"""
