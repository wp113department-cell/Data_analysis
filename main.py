"""
main.py
-------
Streamlit UI for Data Analysis Automation - an agentic workflow.

This file is intentionally "dumb": it only handles page layout, user input,
and displaying results. All the real work happens in:
    - agent/agents.py  (LangChain: LLM calls, prompts, structured output)
    - agent/graph.py   (LangGraph: orchestration of the agent pipeline)
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from agent import agents
from agent.graph import stream_analysis

APP_NAME = "Data Analysis Automation"
APP_TAGLINE = "Agentic Workflow"
DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(
    page_title=f"{APP_NAME} · {APP_TAGLINE}",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLE_QUESTIONS = [
    "Summarize this dataset.",
    "Generate business insights.",
    "Which product generated the highest revenue?",
    "Show monthly revenue.",
    "What are the top five products?",
    "Which customer spent the most?",
    "Which city generated the highest revenue?",
    "Show the distribution of sales.",
    "Are there any missing values?",
]

# node name in agent/graph.py -> (display label, what it does)
PIPELINE_STEPS = {
    "planner": ("Planner", "turned the question into a structured analysis plan"),
    "statistics": ("Statistics", "computed exact summaries and group-bys"),
    "visualization": ("Visualization", "rendered charts in parallel with statistics"),
    "insight": ("Insights", "wrote business insights from the numbers"),
    "report": ("Report", "assembled the final report"),
}

CUSTOM_CSS = """
<style>
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }
[data-testid="stAppDeployButton"], footer { display: none; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #1E1B4B 0%, #312E81 45%, #4F46E5 100%);
    border-radius: 16px;
    padding: 2rem 2.25rem;
    color: #FFFFFF;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px -12px rgba(49, 46, 129, 0.45);
}
.hero-eyebrow {
    display: inline-block;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
    background: rgba(255, 255, 255, 0.14); border: 1px solid rgba(255, 255, 255, 0.22);
    padding: 0.25rem 0.7rem; border-radius: 999px; margin-bottom: 0.9rem;
}
.hero-title { font-size: 2.1rem; font-weight: 700; line-height: 1.15; margin: 0 0 0.5rem 0; }
.hero-title span { color: #A5B4FC; }
.hero-sub { font-size: 1rem; opacity: 0.85; max-width: 760px; margin: 0; }
.hero-tags { margin-top: 1.1rem; display: flex; flex-wrap: wrap; gap: 0.5rem; }
.hero-tag {
    font-size: 0.78rem; font-weight: 500;
    background: rgba(255, 255, 255, 0.10); border-radius: 6px; padding: 0.2rem 0.6rem;
}

/* Section headings */
.section-title { font-size: 1.15rem; font-weight: 600; margin: 0.25rem 0 0.15rem 0; }
.section-sub { font-size: 0.88rem; color: #64748B; margin-bottom: 0.75rem; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
[data-testid="stMetricLabel"] p { font-size: 0.78rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.04em; }
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; color: #0F172A; }

/* Pipeline diagram */
.pipeline { display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; margin: 0.25rem 0 0.5rem 0; }
.pipe-col { display: flex; flex-direction: column; gap: 0.45rem; }
.pipe-node {
    display: flex; align-items: center; gap: 0.5rem;
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px;
    padding: 0.55rem 0.85rem; font-size: 0.86rem; font-weight: 600; color: #334155;
    min-width: 132px;
}
.pipe-node .dot { width: 8px; height: 8px; border-radius: 50%; background: #CBD5E1; flex-shrink: 0; }
.pipe-node.done { border-color: #A5B4FC; background: #EEF2FF; color: #312E81; }
.pipe-node.done .dot { background: #4F46E5; box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.18); }
.pipe-arrow { color: #94A3B8; font-size: 1.1rem; }

/* Feature cards on the empty state */
.feature {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
    padding: 1.1rem 1.2rem; height: 100%;
}
.feature-step { font-size: 0.72rem; font-weight: 700; color: #4F46E5; letter-spacing: 0.08em; text-transform: uppercase; }
.feature-title { font-size: 1rem; font-weight: 600; margin: 0.3rem 0 0.25rem 0; color: #0F172A; }
.feature-text { font-size: 0.87rem; color: #64748B; margin: 0; }

/* Sidebar */
.brand { display: flex; align-items: center; gap: 0.7rem; margin: 0.25rem 0 1.25rem 0; }
.brand-logo {
    width: 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(135deg, #6366F1, #4338CA);
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.95rem; color: #FFFFFF !important;
}
.brand-name { font-weight: 700; font-size: 0.98rem; line-height: 1.2; }
.brand-tag { font-size: 0.75rem; color: #94A3B8 !important; }
.side-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
    color: #94A3B8 !important; margin: 0.4rem 0 0.4rem 0;
}
.status-pill {
    display: flex; align-items: center; gap: 0.5rem;
    background: #1E293B; border: 1px solid #334155; border-radius: 8px;
    padding: 0.5rem 0.7rem; font-size: 0.84rem; margin-bottom: 0.45rem;
}
.status-pill .led { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.led.ok { background: #22C55E; box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.2); }
.led.bad { background: #F59E0B; box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.2); }
.status-pill .k { color: #94A3B8 !important; margin-right: auto; }

/* Answer card */
.answer-q { font-size: 0.8rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 0.25rem; border-bottom: 1px solid #E2E8F0; }
.stTabs [data-baseweb="tab"] { padding: 0.5rem 0.9rem; font-weight: 500; }
</style>
"""


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    defaults = {
        "df": None,
        "file_name": None,
        "result": None,
        "last_chart_dir": None,
        "history": [],  # list of {"question": str, "answer": str}, oldest first
        "load_error": None,
        "uploader_key": 0,
        "question_input": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_overview() -> None:
    """Drop the full pipeline result and its temp chart files."""
    if st.session_state.get("last_chart_dir"):
        shutil.rmtree(st.session_state["last_chart_dir"], ignore_errors=True)
    st.session_state["result"] = None
    st.session_state["last_chart_dir"] = None


def clear_results() -> None:
    clear_overview()
    st.session_state["history"] = []


def reset_session() -> None:
    """Clear everything: uploaded data, results, and any temp chart files."""
    clear_results()
    st.session_state["df"] = None
    st.session_state["file_name"] = None
    st.session_state["load_error"] = None
    st.session_state["question_input"] = ""
    # A new key gives a fresh, empty uploader so the old file isn't re-loaded.
    st.session_state["uploader_key"] += 1


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip stray whitespace/quote characters from column names.

    Real-world CSVs often have headers like `"Game Length"` where a space
    before the quote (`, "Game Length"`) means pandas keeps the space and
    quote marks as literal characters in the column name. Every agent (the
    LLM included) would then have to guess the exact messy literal string,
    which is unreliable - cleaning it once here fixes it everywhere.
    """
    cleaned = [str(c).strip().strip('"').strip("'").strip() for c in df.columns]
    if len(set(cleaned)) == len(cleaned):
        df = df.copy()
        df.columns = cleaned
    return df


def load_csv(source) -> pd.DataFrame:
    """Read and validate a CSV (uploaded file or path). Raises ValueError on problems."""
    try:
        df = pd.read_csv(source)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not read this file as a CSV. Details: {exc}") from exc

    if df.empty or df.shape[1] == 0:
        raise ValueError("The uploaded CSV is empty.")

    df = _clean_column_names(df)

    return df


def set_dataset(source, name: str) -> None:
    """Load a new dataset into the session, discarding results from the previous one."""
    try:
        df = load_csv(source)
    except ValueError as exc:
        st.session_state["load_error"] = str(exc)
        return
    clear_results()
    st.session_state["df"] = df
    st.session_state["file_name"] = name
    st.session_state["load_error"] = None


def _on_upload() -> None:
    uploaded_file = st.session_state.get(f"uploader_{st.session_state['uploader_key']}")
    if uploaded_file is not None:
        set_dataset(uploaded_file, uploaded_file.name)


def _clear_question() -> None:
    st.session_state["question_input"] = ""


def _on_suggestion() -> None:
    choice = st.session_state.get("suggestion")
    if choice:
        st.session_state["question_input"] = choice
    st.session_state["suggestion"] = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="brand">
                <div class="brand-logo">DA</div>
                <div>
                    <div class="brand-name">{APP_NAME}</div>
                    <div class="brand-tag">{APP_TAGLINE}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="side-label">Data source</div>', unsafe_allow_html=True)
        mode = st.segmented_control(
            "Data source",
            ["Upload", "Sample"],
            default="Upload",
            label_visibility="collapsed",
            width="stretch",
        )

        if mode == "Sample":
            samples = sorted(p.name for p in DATA_DIR.glob("*.csv"))
            if samples:
                sample = st.selectbox("Sample dataset", samples, label_visibility="collapsed")
                if st.button("Load sample", icon=":material/download:", width="stretch"):
                    set_dataset(DATA_DIR / sample, sample)
            else:
                st.caption("No sample CSVs found in the data/ folder.")
        else:
            st.file_uploader(
                "Upload CSV",
                type=["csv"],
                key=f"uploader_{st.session_state['uploader_key']}",
                on_change=_on_upload,
                label_visibility="collapsed",
            )

        if st.session_state["load_error"]:
            st.error(st.session_state["load_error"], icon=":material/error:")

        st.markdown('<div class="side-label">Engine</div>', unsafe_allow_html=True)
        key_ok = agents.is_api_key_configured()
        st.markdown(
            f"""
            <div class="status-pill"><span class="led {'ok' if key_ok else 'bad'}"></span>
                <span class="k">Groq API</span><span>{'Connected' if key_ok else 'Key missing'}</span></div>
            <div class="status-pill"><span class="led ok"></span>
                <span class="k">Model</span><span>{agents.GROQ_MODEL_NAME}</span></div>
            <div class="status-pill"><span class="led {'ok' if st.session_state['df'] is not None else 'bad'}"></span>
                <span class="k">Dataset</span><span>{'Loaded' if st.session_state['df'] is not None else 'None'}</span></div>
            """,
            unsafe_allow_html=True,
        )
        if not key_ok:
            st.caption("Add GROQ_API_KEY to a .env file (see .env.example), then restart the app.")

        st.markdown('<div class="side-label">Session</div>', unsafe_allow_html=True)
        if st.button("Reset session", icon=":material/restart_alt:", width="stretch"):
            reset_session()
            st.rerun()

        st.caption("Built with LangChain · LangGraph · Groq · Streamlit")


# ---------------------------------------------------------------------------
# Main page sections
# ---------------------------------------------------------------------------

def section_title(title: str, subtitle: str | None = None) -> None:
    html = f'<div class="section-title">{title}</div>'
    if subtitle:
        html += f'<div class="section-sub">{subtitle}</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-eyebrow">{APP_TAGLINE}</div>
            <div class="hero-title">{APP_NAME.rsplit(" ", 1)[0]} <span>{APP_NAME.rsplit(" ", 1)[1]}</span></div>
            <p class="hero-sub">
                Upload a CSV and ask a question in plain English. A team of AI agents plans the
                analysis, computes statistics, builds charts, and writes insights and a final
                report for you.
            </p>
            <div class="hero-tags">
                <span class="hero-tag">Multi-agent pipeline</span>
                <span class="hero-tag">Parallel execution</span>
                <span class="hero-tag">Exact statistics</span>
                <span class="hero-tag">Downloadable report</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline(done: bool) -> None:
    def node(key: str) -> str:
        cls = "pipe-node done" if done else "pipe-node"
        return f'<div class="{cls}"><span class="dot"></span>{PIPELINE_STEPS[key][0]}</div>'

    arrow = '<span class="pipe-arrow">&rarr;</span>'
    st.markdown(
        f"""
        <div class="pipeline">
            {node("planner")}{arrow}
            <div class="pipe-col">{node("statistics")}{node("visualization")}</div>{arrow}
            {node("insight")}{arrow}{node("report")}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    cols = st.columns(3)
    features = [
        ("Step 1", "Load your data", "Upload a CSV from the sidebar, or pick one of the bundled sample datasets."),
        ("Step 2", "Ask a question", "Type anything in plain English - revenue, trends, top items, missing values."),
        ("Step 3", "Get the analysis", "Receive an answer, statistics, charts, AI insights and a downloadable report."),
    ]
    for col, (step, title, text) in zip(cols, features):
        col.markdown(
            f"""
            <div class="feature">
                <div class="feature-step">{step}</div>
                <div class="feature-title">{title}</div>
                <p class="feature-text">{text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    with st.container(border=True):
        section_title("How the workflow runs", "Statistics and Visualization agents run in parallel; Insights waits for both.")
        render_pipeline(done=False)


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Column": df.columns,
            "Type": [str(t) for t in df.dtypes],
            "Non-null": df.notna().sum().values,
            "Missing": df.isna().sum().values,
            "Missing %": (df.isna().mean() * 100).round(1).values,
            "Unique": df.nunique().values,
        }
    )


def render_data_overview() -> None:
    df: pd.DataFrame = st.session_state["df"]

    section_title("Dataset overview", f"Source: <b>{st.session_state['file_name']}</b>")

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    missing_pct = (missing_cells / total_cells * 100) if total_cells else 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows", f"{df.shape[0]:,}")
    m2.metric("Columns", f"{df.shape[1]:,}")
    m3.metric("Numeric columns", f"{df.select_dtypes('number').shape[1]:,}")
    m4.metric("Missing cells", f"{missing_pct:.1f}%")
    m5.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    preview_tab, profile_tab, summary_tab = st.tabs(
        [":material/table_view: Preview", ":material/view_column: Column profile", ":material/functions: Summary statistics"]
    )
    with preview_tab:
        st.dataframe(df.head(100), width="stretch", height=320)
        st.caption(f"Showing first {min(100, df.shape[0])} of {df.shape[0]:,} rows.")
    with profile_tab:
        st.dataframe(
            column_profile(df),
            width="stretch",
            hide_index=True,
            column_config={
                "Missing %": st.column_config.ProgressColumn(
                    "Missing %", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )
    with summary_tab:
        numeric = df.select_dtypes("number")
        if numeric.empty:
            st.info("This dataset has no numeric columns.")
        else:
            st.dataframe(numeric.describe().T.round(3), width="stretch")


def render_question_form() -> tuple[str | None, str]:
    """Returns (action, question) where action is "analyze", "regenerate" or None."""
    with st.container(border=True):
        section_title("Ask the agents", "Ask anything about your data in plain English.")

        st.pills(
            "Suggestions",
            EXAMPLE_QUESTIONS,
            selection_mode="single",
            key="suggestion",
            on_change=_on_suggestion,
            label_visibility="collapsed",
        )

        question = st.text_area(
            "Your question",
            key="question_input",
            placeholder="e.g. Which product generated the highest revenue?",
            height=96,
            label_visibility="collapsed",
        )

        has_result = st.session_state.get("result") is not None
        col1, col2, col3 = st.columns([2, 2, 1])
        analyze = col1.button(
            "Analyze",
            type="primary",
            icon=":material/play_arrow:",
            width="stretch",
        )
        regenerate = col2.button(
            "Regenerate full overview",
            icon=":material/refresh:",
            width="stretch",
            help="Re-run the full multi-agent pipeline (statistics, charts, insights, report) for this question.",
        )
        col3.button("Clear", icon=":material/backspace:", width="stretch", on_click=_clear_question)

        if has_result:
            st.caption(
                "The overview (statistics, charts, insights, report) was generated from your first question. "
                "New questions update the Answer tab only - use *Regenerate* to rebuild the overview."
            )

    if analyze:
        return "analyze", question
    if regenerate:
        return "regenerate", question
    return None, question


def run_workflow(question: str, full: bool) -> None:
    """Answer the question, and (if `full`) run the whole LangGraph pipeline with live progress."""
    df = st.session_state["df"]
    started = time.perf_counter()

    with st.status("Running agentic workflow...", expanded=True) as status:
        try:
            st.write(":material/forum: **Query agent** - answering your question")
            answer = agents.run_query_agent(question, df)
            st.session_state["history"].append({"question": question, "answer": answer})

            if full:
                clear_overview()
                final_state = None
                for node_name, state in stream_analysis(df, question):
                    st.session_state["last_chart_dir"] = state.get("chart_dir")
                    label, desc = PIPELINE_STEPS.get(node_name, (node_name.title(), "finished"))
                    st.write(f":material/check_circle: **{label}** - {desc}")
                    final_state = state
                st.session_state["result"] = final_state

            elapsed = time.perf_counter() - started
            status.update(label=f"Workflow complete in {elapsed:.1f}s", state="complete", expanded=False)
        except Exception as exc:  # noqa: BLE001
            status.update(label="Workflow failed", state="error", expanded=True)
            st.error(f"Analysis failed: {exc}", icon=":material/error:")


def render_results() -> None:
    history = st.session_state["history"]
    result = st.session_state["result"]

    if not history and not result:
        return

    st.write("")
    section_title("Results")

    if result:
        render_pipeline(done=True)
        if result.get("error"):
            st.warning(f"Part of the pipeline had an issue: {result['error']}", icon=":material/warning:")

    answer_tab, stats_tab, charts_tab, insights_tab, report_tab = st.tabs(
        [
            ":material/forum: Answer",
            ":material/bar_chart: Statistics",
            ":material/monitoring: Charts",
            ":material/lightbulb: AI Insights",
            ":material/description: Final Report",
        ]
    )

    with answer_tab:
        if not history:
            st.info("Ask a question to see an answer here.")
        else:
            latest = history[-1]
            with st.container(border=True):
                st.markdown(f'<div class="answer-q">{_escape(latest["question"])}</div>', unsafe_allow_html=True)
                st.markdown(latest["answer"])
            if len(history) > 1:
                with st.expander(f"Earlier questions ({len(history) - 1})", icon=":material/history:"):
                    for item in reversed(history[:-1]):
                        st.markdown(f"**{item['question']}**")
                        st.markdown(item["answer"])
                        st.divider()

    if not result:
        for tab in (stats_tab, charts_tab, insights_tab, report_tab):
            with tab:
                st.info("Run the full analysis to generate this section.")
        return

    with stats_tab:
        render_statistics(result.get("statistics") or {})

    with charts_tab:
        charts = result.get("charts") or []
        if not charts:
            st.info("No charts were needed to answer this question.")
        cols = st.columns(2)
        for i, chart in enumerate(charts):
            with cols[i % 2].container(border=True):
                st.markdown(f"**{chart['title']}**")
                st.image(chart["path"], width="stretch")

    with insights_tab:
        with st.container(border=True):
            st.markdown(result.get("insights") or "No insights generated.")

    with report_tab:
        report = result.get("report") or ""
        st.download_button(
            "Download report (Markdown)",
            data=report,
            file_name="data_analysis_report.md",
            mime="text/markdown",
            icon=":material/download:",
            type="primary",
            disabled=not report,
        )
        with st.container(border=True):
            st.markdown(report or "No report generated.")


def render_statistics(stats: dict) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", stats.get("row_count", "-"))
    c2.metric("Columns", stats.get("column_count", "-"))
    missing = stats.get("missing_values") or {}
    c3.metric("Columns with missing values", len(missing))

    left, right = st.columns(2)
    with left:
        st.markdown("**Missing values**")
        if missing:
            st.dataframe(
                pd.DataFrame(list(missing.items()), columns=["Column", "Missing"]),
                width="stretch",
                hide_index=True,
            )
        else:
            st.success("No missing values.", icon=":material/check_circle:")
    with right:
        unique_counts = stats.get("categorical_unique_counts") or {}
        if unique_counts:
            st.markdown("**Distinct values per categorical column**")
            st.dataframe(
                pd.DataFrame(list(unique_counts.items()), columns=["Column", "Distinct values"]),
                width="stretch",
                hide_index=True,
            )

    numeric_summary = stats.get("numeric_summary") or {}
    if numeric_summary:
        st.markdown("**Numeric summary**")
        st.dataframe(pd.DataFrame(numeric_summary).T, width="stretch")

    grouped = stats.get("grouped_summary")
    if grouped:
        st.markdown(f"**Grouped by {grouped['groupby_column']} ({grouped['metric_column']})**")
        st.dataframe(pd.DataFrame(grouped["data"]).T, width="stretch")


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

def main() -> None:
    init_session_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_sidebar()
    render_header()

    if st.session_state["df"] is None:
        render_empty_state()
        return

    render_data_overview()
    st.write("")
    action, question = render_question_form()

    if action is not None:
        clean_question = question.strip()
        if not clean_question and action == "regenerate" and st.session_state["history"]:
            clean_question = st.session_state["history"][-1]["question"]

        if not clean_question:
            st.warning("Please type a question first.", icon=":material/edit:")
        elif not agents.is_api_key_configured():
            st.error("GROQ_API_KEY is not set. Add it to a .env file, then restart the app.", icon=":material/key:")
        else:
            full = action == "regenerate" or st.session_state["result"] is None
            run_workflow(clean_question, full=full)

    render_results()


if __name__ == "__main__":
    main()
