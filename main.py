"""
main.py
-------
Streamlit chat UI for Helper Quote - an agentic quoting assistant.

This file only handles the page: sidebar, chat bubbles, live progress and the
quote card. The agent (LangGraph) lives in agent/graph.py, its tools in
agent/quote_tools.py and its memory in agent/memory.py.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from datetime import date
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from agent import agents, memory
from agent.graph import get_graph, run_config
from agent.quote_document import render_quote_pdf
from agent.quote_tools import TOOL_LABELS
from agent.rate_limit import AllModelsBusy
from agent.uploads import jobs as upload_jobs
from agent.uploads.files import save_batch, upload_settings, validate_batch
from agent.sheets import get_workbooks

AVATAR = str(Path(__file__).parent / "assets" / "helper_avatar.svg")
AVATAR_DATA_URI = "data:image/svg+xml;base64," + base64.b64encode(Path(AVATAR).read_bytes()).decode()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="Helper Quote", page_icon=AVATAR, layout="centered")

SUGGESTIONS = [
    "I need a quote for 2 W2136 in white shaker",
    "What colors do you offer?",
    "L-shaped kitchen, 12' and 14' walls - countertop estimate?",
    "I have a quote already - can you update it?",
]

CUSTOM_CSS = """
<style>
:root {
  --ink: #1C0F2D; --plum: #2B1745; --plum-2: #5A4880; --muted: #8A7FA0;
  --cream: #FBF7F1; --sand: #F2F1EF; --lav: #F3EEFF; --lav-2: #D9CFEA;
  --khaki: #FFE07A; --khaki-soft: #FFF3C4; --pink: #FF9BAE; --pink-soft: #FFE4EA;
  --green: #1F7A4D; --mint: #52E0A8;
  --mono: "JetBrains Mono", ui-monospace, monospace;
}
.block-container { padding-top: 1.6rem; padding-bottom: 7rem; max-width: 820px; }
[data-testid="stAppDeployButton"], footer { display: none; }

/* Top bar */
.topbar { display: flex; justify-content: space-between; align-items: flex-end; gap: 1rem; margin-bottom: 1.2rem; }
.eyebrow { font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted); }
.title { font-size: 2.1rem; font-weight: 800; letter-spacing: -0.03em; color: var(--ink); line-height: 1.1; }
.live { display: flex; align-items: center; gap: 0.6rem; }
.live-pill { display: flex; align-items: center; gap: 0.55rem; background: #FFFFFF; border: 1px solid var(--lav-2);
  border-radius: 999px; padding: 0.4rem 0.85rem; font-family: var(--mono); font-size: 0.72rem;
  letter-spacing: 0.14em; color: var(--green); }
.live-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--mint); box-shadow: 0 0 0 0 rgba(82,224,168,.6);
  animation: hq-pulse 1.8s infinite; }
.live-face { width: 40px; height: 40px; border-radius: 50%; background: var(--khaki); display: flex;
  align-items: center; justify-content: center; }
.live-face img { width: 30px; height: 30px; animation: hq-bob 2.6s ease-in-out infinite; }

/* Chat rows */
[data-testid="stChatMessage"] { background: transparent !important; padding: 0.3rem 0; gap: 0.6rem; }
[data-testid="stChatMessageAvatarUser"] { display: none; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { flex-direction: row-reverse; }
[data-testid="stChatMessageAvatarCustom"] { width: 2.5rem; height: 2.5rem; background: transparent;
  animation: hq-bob 2.6s ease-in-out infinite; }
[data-testid="stChatMessageAvatarCustom"] img { width: 100%; height: 100%; }
[data-testid="stChatMessage"]:has(.st-key-thinking_now) [data-testid="stChatMessageAvatarCustom"] {
  animation: hq-jump 0.7s ease-in-out infinite; }

/* Bubbles */
[class*="st-key-umsg"] { background: var(--sand); color: var(--ink); border-radius: 20px 20px 6px 20px;
  padding: 0.7rem 1.05rem; width: fit-content !important; max-width: 80%; margin-left: auto; }
[class*="st-key-amsg"] { background: var(--plum); border-radius: 20px 20px 20px 6px; padding: 0.8rem 1.1rem;
  width: fit-content !important; max-width: 90%; box-shadow: 0 8px 22px -14px rgba(43,23,69,.6); }
[class*="st-key-amsg"] :is(p, li, strong, em, span, h1, h2, h3, h4, td, th, code) { color: #FFFFFF !important; }
[class*="st-key-amsg"] a { color: var(--khaki) !important; }
[class*="st-key-amsg"] [data-testid="stMarkdownContainer"] p, [class*="st-key-umsg"] [data-testid="stMarkdownContainer"] p {
  margin: 0 0 0.45rem 0; }
[class*="st-key-amsg"] [data-testid="stMarkdownContainer"] > :last-child,
[class*="st-key-umsg"] [data-testid="stMarkdownContainer"] > :last-child { margin-bottom: 0; }
[class*="st-key-amsg"] [data-testid="stMarkdownContainer"], [class*="st-key-umsg"] [data-testid="stMarkdownContainer"],
[class*="st-key-quote"] [data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }
[class*="st-key-amsg"], [class*="st-key-umsg"] { overflow: visible; gap: 0 !important; }
.file-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; justify-content: flex-end; margin-top: 0.35rem; }
.file-chip { display: inline-flex; align-items: center; gap: 0.35rem; background: var(--lav); color: var(--plum-2);
  border-radius: 999px; padding: 0.25rem 0.75rem; font-size: 0.82rem; font-weight: 600; }
[class*="st-key-upload_card"] { background: #FFFFFF; border: 1px dashed var(--lav-2); border-radius: 18px; padding: 0.8rem 1rem; }
.chosen { display: inline-block; background: var(--lav); color: var(--plum-2); border-radius: 999px;
  padding: 0.25rem 0.8rem; font-weight: 600; font-size: 0.9rem; }

/* Quote card (khaki, like a booking banner) */
[class*="st-key-quote"] { background: var(--khaki-soft); border: 1.5px solid var(--khaki); border-radius: 20px;
  padding: 1rem 1.15rem; }
.quote-head { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; flex-wrap: wrap; }
.quote-no { font-family: var(--mono); font-size: 0.74rem; font-weight: 500; color: var(--plum-2);
  letter-spacing: 0.14em; text-transform: uppercase; }
.quote-total { font-size: 1.6rem; font-weight: 800; color: var(--ink); }
.quote-meta { font-size: 0.86rem; color: var(--plum-2); }
.quote-badge { display: inline-block; background: var(--khaki); color: var(--ink); font-weight: 700;
  border-radius: 999px; padding: 0.15rem 0.7rem; font-size: 0.78rem; margin-left: 0.4rem; }

/* Thinking / status */
[data-testid="stChatMessage"] [data-testid="stExpander"] details { border: 1px dashed var(--lav-2); background: #FFFFFF80; border-radius: 14px; }
[data-testid="stChatMessage"] [data-testid="stExpander"] summary p { font-size: 0.82rem; color: var(--muted); }

/* Option chips */
[data-testid="stBaseButton-pills"] { background: var(--lav) !important; color: var(--plum-2) !important;
  border: 1px solid transparent !important; font-weight: 600; border-radius: 12px !important; }
[data-testid="stBaseButton-pills"]:hover { border-color: var(--lav-2) !important; color: var(--plum) !important; }
[data-testid="stBaseButton-pillsActive"] { background: var(--plum) !important; color: #FFFFFF !important; border-radius: 12px !important; }

/* Composer */
[data-testid="stBottom"] > div { background: var(--cream); }
[data-testid="stChatInput"], [data-testid="stChatInput"] > div, [data-testid="stChatInputTextArea"] {
  background: #FFFDF8 !important; }
[data-testid="stChatInput"] { border: 1px solid var(--lav-2); border-radius: 16px; box-shadow: 0 6px 20px -14px rgba(43,23,69,.35); }
[data-testid="stChatInputSubmitButton"] { background: var(--plum) !important; color: #FFFFFF !important; border-radius: 999px !important; }

/* Sidebar */
[data-testid="stSidebar"] { border-right: 1px solid var(--sand); }
.brand { display: flex; align-items: center; gap: 0.7rem; margin: 0.2rem 0 1rem 0; padding: 0.7rem 0.8rem;
  border: 1px solid var(--sand); border-radius: 16px; background: var(--cream); }
.brand img { width: 38px; height: 38px; animation: hq-bob 2.6s ease-in-out infinite; }
.brand-name { font-weight: 800; font-size: 0.98rem; line-height: 1.2; color: var(--ink); }
.brand-tag { font-size: 0.76rem; color: var(--green) !important; display: flex; align-items: center; gap: 0.35rem; }
.side-label { font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--muted) !important; margin: 1rem 0 0.4rem 0.2rem; }
[class*="st-key-thread_"] button { background: transparent; border: none; justify-content: flex-start; padding-left: 0.7rem; }
[class*="st-key-thread_"] button > div { justify-content: flex-start; width: 100%; }
[class*="st-key-thread_"] button p { text-align: left; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  font-weight: 600; color: var(--plum); }
[class*="st-key-thread_"] [data-testid="stBaseButton-primary"] { background: var(--lav) !important; }
[class*="st-key-thread_"] button:hover { background: var(--cream); }
[class*="st-key-del_"] button[data-testid="stPopoverButton"] {
  border: none !important; background: transparent !important; color: var(--muted) !important; padding: 0 !important;
  min-height: 2rem; }
[class*="st-key-del_"] button[data-testid="stPopoverButton"]:hover { color: #B3261E !important; }
[class*="st-key-del_"] button[data-testid="stPopoverButton"] > div > div[aria-hidden="true"] { display: none; }  /* no chevron */
[class*="st-key-confirmdel_"] button { background: #B3261E !important; border-color: #B3261E !important; color: #FFFFFF !important; }
[class*="st-key-confirmdel_"] button p { color: #FFFFFF !important; }
.status-pill { display: flex; align-items: center; gap: 0.5rem; background: var(--cream); border: 1px solid var(--sand);
  border-radius: 12px; padding: 0.45rem 0.75rem; font-size: 0.8rem; margin-bottom: 0.4rem; color: var(--plum); }
.status-pill .led { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.led.ok { background: var(--mint); } .led.bad { background: var(--khaki); }
.status-pill .k { color: var(--muted) !important; margin-right: auto; font-family: var(--mono); font-size: 0.72rem; }

@keyframes hq-bob { 0%, 100% { transform: translateY(0) rotate(0deg); } 50% { transform: translateY(-3px) rotate(-3deg); } }
@keyframes hq-jump { 0%, 100% { transform: translateY(0) scale(1); } 40% { transform: translateY(-8px) scale(1.06, .96); }
  60% { transform: translateY(-6px) rotate(4deg); } }
@keyframes hq-pulse { 0% { box-shadow: 0 0 0 0 rgba(82,224,168,.6); } 70% { box-shadow: 0 0 0 7px rgba(82,224,168,0); }
  100% { box-shadow: 0 0 0 0 rgba(82,224,168,0); } }
</style>
"""


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def init_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = st.query_params.get("chat") or uuid.uuid4().hex[:12]
    st.session_state.setdefault("role", "customer")
    st.query_params["chat"] = st.session_state["thread_id"]


def new_chat() -> None:
    st.session_state["thread_id"] = uuid.uuid4().hex[:12]
    st.query_params["chat"] = st.session_state["thread_id"]


def delete_chat(thread_id: str) -> None:
    memory.delete_thread(thread_id)
    if st.session_state.get("thread_id") == thread_id:
        new_chat()


def open_chat(thread_id: str) -> None:
    st.session_state["thread_id"] = thread_id
    st.query_params["chat"] = thread_id


def staff_pin() -> str | None:
    return os.getenv("STAFF_PIN") or None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f'<div class="brand"><img src="{AVATAR_DATA_URI}" alt=""><div>'
            '<div class="brand-name">Helper Quote</div>'
            '<div class="brand-tag"><span class="live-dot"></span>Helper is live</div>'
            "</div></div>",
            unsafe_allow_html=True,
        )
        st.button("New chat", icon=":material/add_comment:", width="stretch", type="primary", on_click=new_chat)

        st.markdown('<div class="side-label">Conversations</div>', unsafe_allow_html=True)
        threads = memory.list_threads()
        if not threads:
            st.caption("No conversations yet.")
        for t in threads:
            current = t["thread_id"] == st.session_state["thread_id"]
            title_col, del_col = st.columns([6, 1], gap=None, vertical_alignment="center")
            title_col.button(
                thread_label(t),
                key=f"thread_{t['thread_id']}",
                width="stretch",
                type="primary" if current else "secondary",
                on_click=open_chat,
                args=(t["thread_id"],),
            )
            with del_col.popover("", icon=":material/close:", key=f"del_{t['thread_id']}", help="Delete this chat"):
                st.markdown("**Delete this chat?**")
                st.caption("Messages and uploaded files are removed. Quotes stay in the customer's history.")
                st.button("Delete", key=f"confirmdel_{t['thread_id']}", type="primary", width="stretch",
                          on_click=delete_chat, args=(t["thread_id"],))

        st.markdown('<div class="side-label">Mode</div>', unsafe_allow_html=True)
        render_mode_switch()

        st.markdown('<div class="side-label">Price sheets (live)</div>', unsafe_allow_html=True)
        pills = []
        for wb in get_workbooks().list_workbooks():
            ok = "error" not in wb
            detail = f"updated {wb['last_updated'].replace('T', ' ')}" if ok else "error"
            pills.append(f'<div class="status-pill"><span class="led {"ok" if ok else "bad"}"></span>'
                         f'<span class="k">{wb["brand"]}</span><span>{detail}</span></div>')
        key_ok = agents.is_api_key_configured()
        pills.append(f'<div class="status-pill"><span class="led {"ok" if key_ok else "bad"}"></span>'
                     f'<span class="k">AI</span><span>{agents.model_name("agent").replace(":", " · ", 1) if key_ok else "Key missing"}</span></div>')
        st.markdown("".join(pills), unsafe_allow_html=True)
        missing = agents.missing_models()
        if missing:
            st.warning("Not available on this Groq account: " + ", ".join(missing)
                       + ". Update models in config/settings.json.", icon=":material/warning:")

        if st.session_state["role"] == "staff":
            render_handoffs()
            render_learned_rules()

        st.caption("LangGraph · LangChain · Groq + OpenRouter · Streamlit")


def thread_label(t: dict) -> str:
    title = t.get("title") or "New conversation"
    status = t.get("status") or ""
    if status == "Needs you":
        return f"{title}  :red[● Needs you]"
    if status.startswith("Quote"):
        return f"{title}  :green[● {status}]"
    return title


def render_handoffs() -> None:
    handoffs = memory.open_handoffs()
    st.markdown(f'<div class="side-label">Needs you ({len(handoffs)})</div>', unsafe_allow_html=True)
    if not handoffs:
        st.caption("No chats waiting for a designer.")
    for h in handoffs:
        with st.container(border=True):
            st.markdown(f"**{':red[URGENT] ' if h['urgency'] == 'urgent' else ''}{h['reason']}**")
            st.caption(h["summary"])
            c1, c2 = st.columns(2)
            c1.button("Open", key=f"ho_open_{h['thread_id']}", width="stretch",
                      on_click=open_chat, args=(h["thread_id"],))
            if c2.button("Resolved", key=f"ho_done_{h['thread_id']}", width="stretch"):
                memory.resolve_handoff(h["thread_id"])
                st.rerun()


def render_mode_switch() -> None:
    pin = staff_pin()
    if st.session_state["role"] == "staff":
        st.success("Staff mode - you can teach rules", icon=":material/badge:")
        if st.button("Switch to customer mode", width="stretch"):
            st.session_state["role"] = "customer"
            st.rerun()
        return
    if not pin:
        st.caption("Customer mode. Set STAFF_PIN in .env to enable staff mode (teach rules, look up customers).")
        return
    with st.popover("Staff login", icon=":material/lock:", width="stretch"):
        entered = st.text_input("Staff PIN", type="password")
        if st.button("Enter staff mode", width="stretch"):
            if entered == pin:
                st.session_state["role"] = "staff"
                st.rerun()
            else:
                st.error("Wrong PIN")


def render_learned_rules() -> None:
    st.markdown('<div class="side-label">Rules taught in chat</div>', unsafe_allow_html=True)
    rules = memory.learned_rules()
    if not rules:
        st.caption('None yet. Teach one in chat: "Remember: espresso goes to JSI".')
    for r in rules:
        c1, c2 = st.columns([5, 1])
        c1.caption(r["rule"])
        if c2.button("", key=f"del_{r['id']}", icon=":material/delete:", help="Forget this rule"):
            memory.delete_learned_rule(r["id"])
            st.rerun()


# ---------------------------------------------------------------------------
# Chat rendering
# ---------------------------------------------------------------------------

def group_turns(messages):
    """Split the message list into (user_message, [assistant-side messages]) turns."""
    turns, current = [], None
    for m in messages:
        if isinstance(m, HumanMessage):
            current = (m, [])
            turns.append(current)
        elif current is not None:
            current[1].append(m)
    return turns


@st.cache_data(show_spinner=False, max_entries=200)
def quote_pdf(quote_number: str) -> bytes:
    """Quotes never change once created, so each PDF is built once."""
    return render_quote_pdf(memory.get_quote(quote_number))


def render_quote_card(quote_number: str) -> None:
    q = memory.get_quote(quote_number)
    if not q:
        return
    t = q["totals"]
    with st.container(key=f"quote_{q['quote_number']}_{uuid.uuid4().hex[:6]}"):
        st.markdown(
            f'<div class="quote-head"><div><div class="quote-no">Quote {q["quote_number"]}<span class="quote-badge">Ready</span></div>'
            f'<div class="quote-meta">{q["customer"]["name"]} · {q["brand_display"]} · <b>{q["line"]}</b> · '
            f'valid until {q["valid_until"]}</div></div>'
            f'<div class="quote-total">${t["grand_total"]:,.2f}</div></div>',
            unsafe_allow_html=True,
        )
        rows = [{"Item #": i["sku"], "Description": i["description"], "Qty": i["quantity"],
                 "Unit price": i["unit_price"], "Total": i["total"]} for i in q["items"]]
        st.dataframe(
            rows, hide_index=True, width="stretch",
            column_config={
                "Unit price": st.column_config.NumberColumn(format="$%.2f"),
                "Total": st.column_config.NumberColumn(format="$%.2f"),
            },
        )
        lines = [f"Cabinets & accessories: **\\${t['subtotal']:,.2f}**"]
        if q["assembled"]:
            lines.append(f"Assembly ({q['assembly_fee_percent']:g}%): **\\${t['assembly_fee']:,.2f}**")
        for x in q["extras"]:
            lines.append(money_safe(f"{x['label']}: **${x['amount']:,.2f}** — {x['detail']}"))
        lines.append("Shipping: quoted separately by the office" if q["shipping"] else "Pickup / no shipping")
        st.markdown("  \n".join(lines))
        if q.get("unpriced_items"):
            st.warning("To be confirmed by a designer: " + ", ".join(q["unpriced_items"]), icon=":material/info:")
        st.download_button(
            "Download quote (PDF)",
            data=quote_pdf(q["quote_number"]),
            file_name=f"{q['quote_number']}.pdf",
            mime="application/pdf",
            icon=":material/picture_as_pdf:",
            type="primary",
            key=f"pdf_{q['quote_number']}_{uuid.uuid4().hex[:6]}",
        )


def render_assistant_turn(parts) -> None:
    thinking, steps, quotes, answer = [], [], [], ""
    questions, clarifications = {}, []
    for m in parts:
        if isinstance(m, AIMessage):
            reasoning = m.additional_kwargs.get("reasoning_content")
            if reasoning:
                thinking.append(reasoning)
            for tc in m.tool_calls:
                steps.append(TOOL_LABELS.get(tc["name"], tc["name"]))
                if tc["name"] == "ask_customer":
                    questions[tc["id"]] = tc["args"].get("question", "")
            if m.content and not m.tool_calls:
                answer = m.content
        elif isinstance(m, ToolMessage) and m.name == "ask_customer" and m.tool_call_id in questions:
            try:
                chosen = json.loads(m.content).get("customer_answer")
            except (TypeError, ValueError):
                chosen = m.content
            clarifications.append((questions[m.tool_call_id], chosen))
        elif isinstance(m, ToolMessage) and m.name == "create_quote":
            try:
                data = json.loads(m.content)
            except (TypeError, ValueError):
                continue
            if data.get("status") == "created":
                quotes.append(data["quote_number"])

    if thinking or steps:
        with st.expander("Thinking & steps", icon=":material/psychology:"):
            for s in steps:
                st.markdown(f":material/check_circle: {s}")
            for text in thinking:
                st.caption(text)
    for question, chosen in clarifications:
        bubble(question)
        picked = ", ".join(chosen) if isinstance(chosen, list) else str(chosen)
        st.markdown(f'<span class="chosen">↳ {html_escape(picked)}</span>', unsafe_allow_html=True)
    if answer:
        bubble(answer)
    for number in quotes:
        render_quote_card(number)


def bubble(text: str, who: str = "a") -> None:
    """A chat bubble: dark plum for the agent ('a'), soft sand for the user ('u')."""
    with st.container(key=f"{who}msg_{uuid.uuid4().hex[:8]}"):
        st.markdown(money_safe(text))


def money_safe(text: str) -> str:
    """Streamlit treats $...$ as LaTeX math - escape dollar signs so prices render as text."""
    return str(text).replace("\\$", "$").replace("$", "\\$")


def html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_pending_question(pending) -> str | list | None:
    """The agent paused (LangGraph interrupt) to ask the customer: show clickable options."""
    ask = pending.value or {}
    options = ask.get("options") or []
    key = f"ask_{getattr(pending, 'id', '')}"
    with st.chat_message("assistant", avatar=AVATAR):
        bubble(ask.get("question", "Could you clarify?"))
        if not options:
            st.caption("Type your answer below.")
            return None
        if ask.get("allow_multiple"):
            picked = st.pills("Options", options, selection_mode="multi", key=key, label_visibility="collapsed")
            if st.button("Send", icon=":material/send:", key=f"{key}_send", disabled=not picked):
                return picked
            st.caption("Pick one or more, or type a different answer below.")
            return None
        picked = st.pills("Options", options, key=key, label_visibility="collapsed")
        st.caption("Pick one, or type a different answer below.")
        return picked


@st.cache_data(show_spinner=False, max_entries=300)
def upright_thumbnail(path: str) -> bytes:
    """Phone photos are often stored sideways with a rotation tag - show them upright."""
    from io import BytesIO

    from PIL import Image, ImageOps

    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((330, 330))
        out = BytesIO()
        img.save(out, format="JPEG", quality=85)
        return out.getvalue()


def render_user_message(text: str, files: list | None = None) -> None:
    """User bubble + attachment chips (image thumbnails / PDF chips)."""
    if text.strip():
        bubble(text, "u")
    if files:
        chips = "".join(
            f'<span class="file-chip">{"📄" if f["kind"] == "pdf" else "🖼️"} {html_escape(f["name"])}'
            f'{" · " + str(f.get("pages", 1)) + " pages" if f["kind"] == "pdf" else ""}</span>'
            for f in files)
        st.markdown(f'<div class="file-chips">{chips}</div>', unsafe_allow_html=True)
        images = [f for f in files if f["kind"] == "image" and Path(f.get("path", "")).exists()]
        if images:
            cols = st.columns([1] * len(images) + [max(1, 6 - len(images))])
            for col, f in zip(cols[: len(images)], images):
                col.image(upright_thumbnail(f["path"]), width=110)


def render_history(messages) -> None:
    for human, parts in group_turns(messages):
        with st.chat_message("user"):
            upload = human.additional_kwargs.get("upload") or {}
            render_user_message(human.additional_kwargs.get("visible_text", human.content), upload.get("files"))
        if parts:
            with st.chat_message("assistant", avatar=AVATAR):
                render_assistant_turn(parts)


def render_suggestions(messages) -> str | None:
    """Clickable next steps the understanding step composed for its last reply."""
    last = messages[-1]
    suggestions = last.additional_kwargs.get("suggestions") if isinstance(last, AIMessage) else None
    if not suggestions:
        return None
    return st.pills("Next steps", suggestions, label_visibility="collapsed", key=f"next_{last.id}")


def render_welcome() -> str | None:
    with st.chat_message("assistant", avatar=AVATAR):
        bubble(
            "Hi! I'm **Helper Quote**. Send me your cabinet list (item codes and quantities) and the "
            "color you want, and I'll put together a quote right away. I can also estimate countertops, "
            "handles and installation."
        )
    return st.pills("Try", SUGGESTIONS, label_visibility="collapsed", key=f"sugg_{st.session_state['thread_id']}")


def run_turn(text, resume: bool = False, upload: dict | None = None, show_user: bool = True,
             continue_run: bool = False) -> None:
    """Send a new message (optionally with read attachments), answer a pending question (resume=True),
    or finish a run that was cut off, e.g. the tab was closed mid-reply (continue_run=True)."""
    graph = get_graph()
    config = run_config(st.session_state["thread_id"], st.session_state["role"])
    shown = ", ".join(text) if isinstance(text, list) else str(text)
    thread = memory.get_thread(st.session_state["thread_id"])
    memory.update_thread(st.session_state["thread_id"], title=thread.get("title") or (shown or "Uploaded files")[:60])
    if continue_run:
        graph_input = None  # LangGraph picks up from the last saved step
        show_user = False
    elif resume:
        graph_input = Command(resume=text)
    elif upload:
        names = ", ".join(f["name"] for f in upload["files"])
        note = (f"[ATTACHED FILES: {names} | upload id: {upload['upload_id']} | "
                f"reading {upload.get('state', 'done')}. Use read_uploaded_files.]")
        graph_input = {"messages": [HumanMessage(
            f"{shown}\n\n{note}".strip(),
            additional_kwargs={"visible_text": shown, "upload": {"id": upload["upload_id"], "files": upload["files"]}})]}
    else:
        graph_input = {"messages": [HumanMessage(shown)]}

    if show_user:
        with st.chat_message("user"):
            render_user_message(shown, (upload or {}).get("files"))
    with st.chat_message("assistant", avatar=AVATAR):
        started = time.perf_counter()
        st.container(key="thinking_now").empty()
        with st.status("Thinking...", expanded=False) as status:
            try:
                for mode, chunk in graph.stream(graph_input, config,
                                                stream_mode=["updates", "custom"]):
                    if mode == "custom" and "waiting_seconds" in (chunk or {}):
                        status.update(label=f"Waiting {chunk['waiting_seconds']}s for the Groq rate limit...")
                        continue
                    for node, output in (chunk or {}).items():
                        if node == "agent":
                            for m in (output or {}).get("messages", []):
                                for tc in getattr(m, "tool_calls", []) or []:
                                    status.update(label=TOOL_LABELS.get(tc["name"], tc["name"]) + "...")
                                    st.write(f":material/check_circle: {TOOL_LABELS.get(tc['name'], tc['name'])}")
                        elif node == "understand":
                            status.update(label="Understanding your message...")
                        elif node == "remember":
                            status.update(label="Saving to memory...")
                status.update(label=f"Done in {time.perf_counter() - started:.1f}s", state="complete")
            except AllModelsBusy as exc:
                status.update(label="Usage limit reached", state="error", expanded=False)
                minutes = max(1, round(exc.retry_after / 60))
                bubble(f"I've reached today's AI usage limit, so I can't finish this right now. "
                       f"Please try again in about **{minutes} minute{'s' if minutes != 1 else ''}** - "
                       "everything you told me is saved, so we can pick up right where we left off.")
                if st.session_state["role"] == "staff":
                    st.caption("Staff: Groq daily token limit reached on all configured models. "
                               "Upgrade to Groq Dev tier or add fallback models in config/settings.json.")
                return
            except Exception as exc:  # noqa: BLE001
                logging.getLogger("helper_quote.app").exception(
                    "turn failed (thread %s)", st.session_state["thread_id"])
                status.update(label="Something went wrong", state="error", expanded=False)
                bubble("Sorry, something went wrong on my side while working on that. "
                       "Please send your message again - nothing you told me was lost.")
                if st.session_state["role"] == "staff":
                    st.caption(f"Staff detail: {type(exc).__name__}: {str(exc)[:300]}")
                return
    st.rerun()


# ---------------------------------------------------------------------------
# Uploads (read in the background by Celery workers / a worker thread)
# ---------------------------------------------------------------------------

def start_upload(text: str, uploaded_files) -> None:
    """Validate + store the attachments and queue them for the background reader."""
    tid = st.session_state["thread_id"]
    accepted, problems = validate_batch([(f.name, f.getvalue()) for f in uploaded_files])
    st.session_state["upload_problems"] = problems
    if not accepted:
        if text.strip():
            run_turn(text)
        return
    batch = save_batch(tid, accepted)
    mode = upload_jobs.submit(batch)
    memory.update_thread(tid, pending_upload={
        "upload_id": batch.batch_id, "text": text, "mode": mode,
        "files": [{"name": f.name, "kind": f.kind, "pages": f.pages, "path": f.path} for f in batch.files]})
    st.rerun()


@st.fragment(run_every=1.5)
def upload_progress(tid: str, upload_id: str) -> None:
    s = upload_jobs.status(tid, upload_id)
    if s.get("state") in ("done", "error"):
        st.rerun(scope="app")  # hand the result to the agent
    with st.container(key=f"upload_card_{upload_id}"):
        st.markdown("**Reading your files…** you can keep chatting meanwhile.")
        st.progress(int(s.get("progress", 0)) / 100, text=s.get("status", "Waiting to start"))


def render_pending_upload(pending_question) -> bool:
    """Show an upload that is still being read. When it's done, continue the conversation."""
    tid = st.session_state["thread_id"]
    pu = memory.get_thread(tid).get("pending_upload") or {}
    if not pu or pending_question is not None:
        return False
    state = upload_jobs.status(tid, pu["upload_id"]).get("state")
    with st.chat_message("user"):
        render_user_message(pu.get("text", ""), pu["files"])
    if state in ("done", "error"):
        memory.update_thread(tid, pending_upload={})
        run_turn(pu.get("text", ""), upload={**pu, "state": state}, show_user=False)
        return True
    with st.chat_message("assistant", avatar=AVATAR):
        st.container(key="thinking_now").empty()
        upload_progress(tid, pu["upload_id"])
    return True


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main() -> None:
    init_session()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_sidebar()

    st.markdown(
        f'<div class="topbar"><div><div class="eyebrow">{date.today().strftime("%A %d %B")}</div>'
        '<div class="title">Helper Quote</div></div>'
        '<div class="live"><div class="live-pill"><span class="live-dot"></span>HELPER ON</div>'
        f'<div class="live-face"><img src="{AVATAR_DATA_URI}" alt=""></div></div></div>',
        unsafe_allow_html=True,
    )

    if not agents.is_api_key_configured():
        st.error("GROQ_API_KEY is not set. Add it to a .env file, then restart the app.", icon=":material/key:")
        return

    state = get_graph().get_state(run_config(st.session_state["thread_id"]))
    messages = state.values.get("messages", []) if state and state.values else []

    pending = state.interrupts[0] if state and state.interrupts else None

    # A reply that was cut off (tab closed / connection lost mid-run): finish it from the checkpoint, once.
    unfinished = bool(state and state.next and not state.interrupts)
    resumed_key = f"resumed_{st.session_state['thread_id']}_{state.config['configurable'].get('checkpoint_id') if state else ''}"
    if unfinished and not st.session_state.get(resumed_key):
        st.session_state[resumed_key] = True
        if messages:
            render_history(messages)
        st.caption(":material/replay: Finishing your last request...")
        run_turn("", continue_run=True)
        return

    picked = None
    upload_waiting = bool(memory.get_thread(st.session_state["thread_id"]).get("pending_upload"))
    if not messages and not upload_waiting:
        picked = render_welcome()
    else:
        if state.values.get("summary"):
            st.caption(":material/history: Earlier messages were summarized into memory.")
        render_history(messages)
    if pending is not None:
        picked = render_pending_question(pending)
    reading = render_pending_upload(pending)

    for problem in st.session_state.pop("upload_problems", []) or []:
        st.warning(problem, icon=":material/attach_file_off:")

    cfg = upload_settings()
    max_chars = int(agents.load_settings().get("chat", {}).get("max_message_chars", 4000))
    value = st.chat_input(
        "Type your answer..." if pending else "Type your cabinet list, or attach a layout / photo / PDF...",
        max_chars=max_chars,
        accept_file=False if pending else "multiple",
        file_type=None if pending else ["pdf"] + list(cfg["image_types"]),
    )
    prompt, files = (value, []) if (value is None or isinstance(value, str)) else (value.text or "", list(value.files))
    if files:
        start_upload((prompt or "")[:max_chars], files)
        return
    if prompt and not prompt.strip():
        prompt = None
    if prompt and len(prompt) > max_chars:  # defence in depth (pasted text, API clients)
        prompt = prompt[:max_chars]
    if reading and not prompt:
        return
    if not pending and picked is None and messages:
        picked = render_suggestions(messages)
    text = prompt or picked
    if text:
        run_turn(text, resume=pending is not None)


if __name__ == "__main__":
    main()
