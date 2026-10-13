"""
agents.py
---------
This file holds ALL the LangChain-related code for the AI Data Analyst Agent.

LangChain is used ONLY for:
    - Talking to the Groq LLM (openai/gpt-oss-120b)
    - Prompt templates
    - Structured output (turning free text into a validated Python object)
    - Output parsing

LangChain is NOT used for orchestration. The graph/flow control lives in
graph.py (that part is LangGraph's job).

This file also contains the pure-Python "Statistics Agent" and
"Visualization Agent" logic. Those two agents do not need an LLM at all -
they just use pandas and matplotlib.
"""

from __future__ import annotations

import ast
import os
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, required for Streamlit/servers
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

GROQ_MODEL_NAME = "openai/gpt-oss-120b"
ALLOWED_CHART_TYPES = ["bar", "histogram", "pie", "scatter"]


# ---------------------------------------------------------------------------
# LLM setup (LangChain)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.2) -> ChatGroq:
    """
    Create (and cache) the LangChain ChatGroq LLM client.

    Raises:
        ValueError: if GROQ_API_KEY is missing from the environment/.env file.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to a .env file in the project "
            "root (see .env.example)."
        )
    return ChatGroq(model=GROQ_MODEL_NAME, api_key=api_key, temperature=temperature)


def is_api_key_configured() -> bool:
    """Quick check used by the UI to show API status without raising."""
    api_key = os.getenv("GROQ_API_KEY")
    return bool(api_key) and api_key != "your_key_here"


# ---------------------------------------------------------------------------
# Planner Agent (LangChain: structured output)
# ---------------------------------------------------------------------------

class AnalysisPlan(BaseModel):
    """Structured plan produced by the Planner Agent."""

    intent_summary: str = Field(
        description="One short sentence describing what the user wants to know."
    )
    relevant_columns: List[str] = Field(
        default_factory=list,
        description="Dataset column names that are relevant to answering the question.",
    )
    groupby_column: Optional[str] = Field(
        default=None,
        description="A categorical column to group by, if grouping helps answer the question. Otherwise null.",
    )
    metric_column: Optional[str] = Field(
        default=None,
        description="A numeric column to analyze/aggregate, if relevant. Otherwise null.",
    )
    chart_types: List[str] = Field(
        default_factory=list,
        description=(
            "Which chart types would help answer the question. "
            "Choose only from: bar, histogram, pie, scatter. Empty list if no chart is needed."
        ),
    )
    needs_grouped_summary: bool = Field(
        default=False,
        description="True if a group-by statistical summary (sum/mean/count per group) is needed.",
    )


PLANNER_SYSTEM_PROMPT = """You are the Planner Agent inside an AI Data Analyst system.
You will be given the columns of a pandas DataFrame and a user's question.
Your job is to decide what needs to be computed to answer the question well.

Rules:
- Only use column names that actually exist in the dataset.
- chart_types must only contain values from: bar, histogram, pie, scatter.
- If the question is a general request like "summarize this dataset", pick a
  sensible default groupby/metric column and a couple of useful chart types.
- Keep intent_summary to one sentence.
"""


def run_planner_agent(user_question: str, df: pd.DataFrame) -> AnalysisPlan:
    """Ask the LLM to turn the user's question into a structured analysis plan."""
    llm = get_llm()
    structured_llm = llm.with_structured_output(AnalysisPlan)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PLANNER_SYSTEM_PROMPT),
            (
                "human",
                "Dataset columns and types:\n{columns}\n\n"
                "Sample rows:\n{sample}\n\n"
                "User question: {question}",
            ),
        ]
    )

    columns_desc = "\n".join(f"- {c} ({str(df[c].dtype)})" for c in df.columns)
    sample = df.head(3).to_string()

    chain = prompt | structured_llm
    plan: AnalysisPlan = chain.invoke(
        {"columns": columns_desc, "sample": sample, "question": user_question}
    )

    # Defensive cleanup: drop any hallucinated column names / chart types.
    valid_columns = set(df.columns)
    plan.relevant_columns = [c for c in plan.relevant_columns if c in valid_columns]
    if plan.groupby_column not in valid_columns:
        plan.groupby_column = None
    if plan.metric_column not in valid_columns:
        plan.metric_column = None
    plan.chart_types = [c for c in plan.chart_types if c in ALLOWED_CHART_TYPES]

    return plan


# ---------------------------------------------------------------------------
# Statistics Agent (pure pandas - no LLM needed)
# ---------------------------------------------------------------------------

def _pick_fallback_column(df: pd.DataFrame, dtype_kind: str) -> Optional[str]:
    """Pick a reasonable default column if the planner didn't provide one."""
    if dtype_kind == "numeric":
        cols = df.select_dtypes(include="number").columns.tolist()
    else:
        cols = df.select_dtypes(include=["object", "category", "str", "string"]).columns.tolist()
    return cols[0] if cols else None


def compute_statistics(df: pd.DataFrame, plan: AnalysisPlan) -> Dict[str, Any]:
    """
    Compute dataset statistics: shape, missing values, numeric summary,
    and (if relevant) a grouped summary based on the planner's output.
    """
    stats: Dict[str, Any] = {}

    stats["row_count"] = int(df.shape[0])
    stats["column_count"] = int(df.shape[1])
    stats["columns"] = list(df.columns)

    missing = df.isna().sum()
    stats["missing_values"] = {c: int(missing[c]) for c in df.columns if missing[c] > 0}

    numeric_df = df.select_dtypes(include="number")
    numeric_summary = {}
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.empty:
            continue
        numeric_summary[col] = {
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
        }
    stats["numeric_summary"] = numeric_summary

    categorical_df = df.select_dtypes(include=["object", "category", "str", "string"])
    categorical_counts = {}
    categorical_unique_counts = {}
    for col in categorical_df.columns:
        top_values = df[col].value_counts().head(5)
        categorical_counts[col] = {str(k): int(v) for k, v in top_values.items()}
        categorical_unique_counts[col] = int(df[col].nunique(dropna=True))
    stats["categorical_top_values"] = categorical_counts
    # Exact unique-value counts (top_values above only shows the top 5, which is
    # NOT the same as the total distinct count - this field is what answers
    # "how many X are there" questions accurately.
    stats["categorical_unique_counts"] = categorical_unique_counts

    groupby_col = plan.groupby_column or _pick_fallback_column(df, "categorical")
    metric_col = plan.metric_column or _pick_fallback_column(df, "numeric")

    if plan.needs_grouped_summary and groupby_col and metric_col and groupby_col in df.columns and metric_col in df.columns:
        grouped = (
            df.groupby(groupby_col)[metric_col]
            .agg(["sum", "mean", "count"])
            .round(2)
            .sort_values("sum", ascending=False)
            .head(10)
        )
        stats["grouped_summary"] = {
            "groupby_column": groupby_col,
            "metric_column": metric_col,
            "data": grouped.to_dict(orient="index"),
        }

    return stats


# ---------------------------------------------------------------------------
# Visualization Agent (pure matplotlib - no LLM needed)
# ---------------------------------------------------------------------------

def create_charts(df: pd.DataFrame, plan: AnalysisPlan, output_dir: str) -> List[Dict[str, str]]:
    """
    Create only the charts relevant to the question, based on the planner's
    chart_types list. Returns a list of {"title": ..., "path": ...} dicts.
    """
    os.makedirs(output_dir, exist_ok=True)
    charts: List[Dict[str, str]] = []

    groupby_col = plan.groupby_column or _pick_fallback_column(df, "categorical")
    metric_col = plan.metric_column or _pick_fallback_column(df, "numeric")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    def save_fig(fig, name: str) -> str:
        path = os.path.join(output_dir, f"{name}_{uuid.uuid4().hex[:8]}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    if "bar" in plan.chart_types and groupby_col and metric_col:
        grouped = df.groupby(groupby_col)[metric_col].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(7, 4))
        grouped.plot(kind="bar", ax=ax, color="#4C72B0")
        ax.set_title(f"{metric_col} by {groupby_col}")
        ax.set_ylabel(metric_col)
        ax.set_xlabel(groupby_col)
        plt.xticks(rotation=30, ha="right")
        charts.append({"title": f"{metric_col} by {groupby_col} (Bar Chart)", "path": save_fig(fig, "bar")})

    if "histogram" in plan.chart_types and metric_col:
        fig, ax = plt.subplots(figsize=(7, 4))
        df[metric_col].dropna().plot(kind="hist", bins=15, ax=ax, color="#55A868")
        ax.set_title(f"Distribution of {metric_col}")
        ax.set_xlabel(metric_col)
        charts.append({"title": f"Distribution of {metric_col} (Histogram)", "path": save_fig(fig, "hist")})

    if "pie" in plan.chart_types and groupby_col:
        counts = df[groupby_col].value_counts().head(6)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%", startangle=90)
        ax.set_title(f"Share of {groupby_col}")
        charts.append({"title": f"Share of {groupby_col} (Pie Chart)", "path": save_fig(fig, "pie")})

    if "scatter" in plan.chart_types and len(numeric_cols) >= 2:
        x_col, y_col = numeric_cols[0], numeric_cols[1]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(df[x_col], df[y_col], color="#C44E52", alpha=0.7)
        ax.set_title(f"{y_col} vs {x_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        charts.append({"title": f"{y_col} vs {x_col} (Scatter Plot)", "path": save_fig(fig, "scatter")})

    return charts


# ---------------------------------------------------------------------------
# Insight Agent (LangChain: prompt template + LLM + output parser)
# ---------------------------------------------------------------------------

INSIGHT_SYSTEM_PROMPT = """You are a friendly senior data analyst.
Explain findings from a dataset in simple, plain English for a business audience.
Be specific and use the numbers you were given - do not invent numbers.

Structure your answer as short markdown bullet points covering:
- Key findings that answer the user's question
- Any notable trends
- Any anomalies or things that stand out
- 2-3 practical business insights or recommendations

Keep it concise (under 200 words).
"""


def run_insight_agent(user_question: str, statistics: Dict[str, Any]) -> str:
    """Use the LLM to turn raw statistics into plain-English insights."""
    llm = get_llm(temperature=0.4)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", INSIGHT_SYSTEM_PROMPT),
            (
                "human",
                "User question: {question}\n\nStatistics (JSON-like):\n{stats}",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": user_question, "stats": statistics})


# ---------------------------------------------------------------------------
# Report Agent (LangChain: prompt template + LLM for the summary paragraph)
# ---------------------------------------------------------------------------

REPORT_SYSTEM_PROMPT = """You are a data analyst writing the executive summary
of a short report. Given the user's question and the insights already
generated, write ONE short paragraph (3-4 sentences) that introduces the
report. Do not repeat the bullet points verbatim, just summarize the story.
"""


def _format_statistics_markdown(stats: Dict[str, Any]) -> str:
    lines = ["| Metric | Value |", "|---|---|"]
    lines.append(f"| Rows | {stats.get('row_count', '-')} |")
    lines.append(f"| Columns | {stats.get('column_count', '-')} |")

    missing = stats.get("missing_values") or {}
    missing_str = ", ".join(f"{k}: {v}" for k, v in missing.items()) if missing else "None"
    lines.append(f"| Missing values | {missing_str} |")

    md = "\n".join(lines)

    unique_counts = stats.get("categorical_unique_counts") or {}
    if unique_counts:
        md += "\n\n**Distinct value counts (exact)**\n\n"
        md += "| Column | Unique Values |\n|---|---|\n"
        for col, count in unique_counts.items():
            md += f"| {col} | {count} |\n"

    numeric_summary = stats.get("numeric_summary") or {}
    if numeric_summary:
        md += "\n\n**Numeric Summary**\n\n"
        md += "| Column | Mean | Median | Min | Max |\n|---|---|---|---|---|\n"
        for col, s in numeric_summary.items():
            md += f"| {col} | {s['mean']} | {s['median']} | {s['min']} | {s['max']} |\n"

    grouped = stats.get("grouped_summary")
    if grouped:
        md += f"\n\n**Grouped Summary: {grouped['metric_column']} by {grouped['groupby_column']}**\n\n"
        md += "| Group | Sum | Mean | Count |\n|---|---|---|---|\n"
        for group_name, row in grouped["data"].items():
            md += f"| {group_name} | {row['sum']} | {row['mean']} | {row['count']} |\n"

    return md


def run_report_agent(
    user_question: str,
    statistics: Dict[str, Any],
    insights: str,
    charts: List[Dict[str, str]],
) -> str:
    """
    Combine statistics, chart titles, and insights into one clean markdown
    report. The opening executive summary paragraph is generated by the LLM;
    the rest is assembled deterministically from real numbers so nothing is
    hallucinated.
    """
    llm = get_llm(temperature=0.3)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REPORT_SYSTEM_PROMPT),
            ("human", "User question: {question}\n\nInsights:\n{insights}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    executive_summary = chain.invoke({"question": user_question, "insights": insights})

    chart_list = "\n".join(f"- {c['title']}" for c in charts) if charts else "No charts were needed for this question."

    report = f"""# Data Analysis Report

**Question:** {user_question}

## Executive Summary
{executive_summary}

## Key Statistics
{_format_statistics_markdown(statistics)}

## Charts Generated
{chart_list}

## AI Insights
{insights}
"""
    return report


# ---------------------------------------------------------------------------
# Query / Response Agent (LangChain: generates + phrases a grounded answer)
# ---------------------------------------------------------------------------
#
# The agents above (statistics/insight/report) only ever look at a
# *summary* of the data (top-5 values, aggregate means, etc). That's why a
# question like "how many countries are there" or "tell me about Nepal"
# used to get a wrong/hallucinated answer - the LLM never saw the real
# number, it just guessed from a truncated top-5 list or made something up.
#
# This agent instead asks the LLM to write a small pandas expression, runs
# that expression against the FULL real dataframe (not a sample), and only
# then asks the LLM to phrase the already-computed, real result into a
# sentence. The LLM never invents numbers - it only reads them back.

_NOT_FOUND_MESSAGE = "I couldn't find anything related to that question in the CSV data."

_MAX_UNIQUE_VALUES_LISTED = 40
_MAX_RESULT_ROWS = 15

# Attribute/function names that must never be reachable from a generated
# expression, even indirectly (file I/O, process/OS access, code execution).
_BLOCKED_ATTR_NAMES = {
    "eval", "exec", "compile", "open", "system", "popen", "subprocess", "os", "sys",
    "__import__", "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "load", "loads", "read_pickle", "to_pickle", "read_sql", "to_sql", "read_csv",
    "to_csv", "read_excel", "to_excel", "read_json", "to_json", "read_html",
    "read_parquet", "to_parquet", "read_feather", "to_feather", "to_clipboard",
    "read_clipboard",
}

_ALLOWED_AST_NODES = (
    ast.Expression, ast.Name, ast.Attribute, ast.Subscript, ast.Call,
    ast.Constant, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.List, ast.Tuple, ast.Dict, ast.Slice, ast.Load, ast.keyword,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
    ast.BitAnd, ast.BitOr, ast.BitXor,
    ast.And, ast.Or, ast.Not, ast.Invert, ast.USub, ast.UAdd,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    # Lambdas are allowed (e.g. `.apply(lambda x: ...)`) - their parameter
    # names are locally bound, not free identifiers, so they're validated
    # with scope tracking in _SafeExpressionValidator below.
    ast.Lambda, ast.arguments, ast.arg,
)

# Safe, side-effect-free builtins the generated expression is allowed to call
# directly (e.g. `float(...)` to cast the "value" column before summing it).
_SAFE_BUILTINS = {
    "int": int, "float": float, "str": str, "bool": bool, "len": len,
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "sorted": sorted, "list": list, "tuple": tuple, "dict": dict, "set": set,
}


class QueryPlan(BaseModel):
    """A single pandas expression that computes the answer to the user's question."""

    pandas_expression: str = Field(
        description=(
            "One single-line Python expression using only `df` and `pd` that "
            "computes the exact answer. Return the exact text NOT_FOUND if the "
            "question cannot be answered from the given columns/values."
        )
    )


QUERY_SYSTEM_PROMPT = """You are a precise data-query agent.
You are given the exact column names, dtypes, and distinct-value info of a
pandas DataFrame called `df`, and a user's question in plain English.

Write ONE single-line Python expression - using only `df` and `pd` - that,
when evaluated with Python's eval(), computes the exact answer.

Rules:
- Only use column names that are listed below. Never invent a column name.
- Combine multiple filter conditions with `&` / `|` (not `and`/`or`), and wrap
  each condition in parentheses, e.g. df[(df['a'] == 1) & (df['b'] == 2)]
- If the question asks for MULTIPLE related pieces of information at once
  (e.g. "which record has the highest X, and what is that X", "which game had
  the longest length and how long was it"), return a small row/slice
  containing ALL the requested columns together - not just one scalar. e.g.
  use df.loc[[df['Length'].idxmax()], ['GameId', 'Length']] instead of only
  df['Length'].max(), so nothing the user asked for is silently dropped.
- "how many DISTINCT/UNIQUE X" (categories) -> df['X'].nunique()
- "how many rows/records match ..." (literally counting rows) -> .count() / .shape[0]
- "total X", "number/amount of X recorded", "how much X", "average X", or any
  question about a metric/quantity itself (income, profit, sales, units,
  employees, activity units, etc.) -> SUM (or mean/max/etc) the actual numeric
  VALUE column for the matching rows. Do NOT use count() for these - count()
  only tells you how many rows matched, not the real total/amount. This is
  especially true for "long format" tables where one column names a metric
  (e.g. "variable") and another column holds its number (e.g. "value") - filter
  rows to that metric, then aggregate the value column, never count rows.
- Filtering by a text value the user mentions (a country, city, product, name, etc.)
  -> always match case-insensitively and allow partial matches, e.g.
  df[df['Country'].astype(str).str.lower().str.contains('nepal', na=False)]
- Never assume the user's capitalization/spelling is exactly what's stored - always use .str.lower().str.contains(...) for text matching.
- If a text column has a "`<col>_numeric`" companion listed below, ALWAYS use
  that companion column for sum/mean/max/min/sort/comparison - it is already
  safely converted (non-numeric entries are NaN, automatically skipped). Never
  sort or compare a str/object column's raw text as if it were numbers (text
  sorts alphabetically, e.g. "9" > "800", giving a wrong "highest" answer).
- If a numeric-looking text column has NO `_numeric` companion, convert it
  yourself first with pd.to_numeric(..., errors='coerce') before any numeric
  operation - do not use df['col'].astype(float), it crashes on non-numeric text.
- Look carefully at each column's full list of unique values below. If one
  value in a breakdown/category column clearly represents "everything in this
  breakdown combined" (e.g. "All Industries" sitting among individual industry
  names, or "Grand_Total"/"Industry_Total" sitting among specific size bands),
  while the other values are genuine narrower parts of that same breakdown,
  then for an overall/combined figure filter TO that one combined value
  instead of summing across the narrower parts - summing both double-counts.
  Do NOT apply this to a column whose values are simply distinct named items
  that happen to include the word "total" as part of their own name (e.g. a
  metric literally called "Total income" sitting alongside other differently
  named metrics like "Activity unit" - that is not an aggregate-of-the-others,
  it is its own standalone metric, so filter to it directly, don't skip it).
  A dataset can have more than one such breakdown column at once (e.g. both an
  industry breakdown AND a business-size breakdown) - if so, filter each of
  them to its own combined/total value at the same time (combine with `&`).
- Columns marked "(mostly empty)" below are formatting leftovers in the source
  file, not real data - never use them unless the question explicitly matches
  their described content.
- You may use these safe builtins if needed: int, float, str, bool, len, abs, round, min, max, sum, sorted.
- A simple `lambda` is allowed inside calls like `.apply(...)` if genuinely needed, but prefer the pre-convert pattern above when converting types.
- If the question is about something clearly NOT present in the columns or data at all, return exactly: NOT_FOUND
- Do not write imports, assignments, comprehensions, or multiple statements - one expression only.
- Do not call file, system, or I/O functions. Only pandas/data operations on `df`.
- Return ONLY the expression text, nothing else.
"""

_SPARSE_COLUMN_THRESHOLD = 0.95
_NUMERIC_LIKE_THRESHOLD = 0.5
_NUMERIC_COMPANION_SUFFIX = "_numeric"

def _augment_numeric_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `<col>_numeric` companion column for any text column that is mostly
    numeric (e.g. a "value" column stored as text only because of an
    occasional non-numeric placeholder like 'C' for confidential/suppressed).

    This means the query agent can aggregate/sort/compare real numbers
    directly via the companion column, instead of writing its own inline
    type-conversion logic - which it tends to get wrong (e.g. crashing on the
    placeholder, or sorting the raw text alphabetically).
    """
    augmented = df.copy(deep=False)
    for col in df.columns:
        if df[col].dtype != object and str(df[col].dtype) != "str":
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        numeric = pd.to_numeric(non_null, errors="coerce")
        parseable_frac = float(numeric.notna().mean())
        if _NUMERIC_LIKE_THRESHOLD <= parseable_frac < 1.0:
            augmented[f"{col}{_NUMERIC_COMPANION_SUFFIX}"] = pd.to_numeric(df[col], errors="coerce")
    return augmented


_AGGREGATE_MATCH_TOLERANCE = 0.02
_AGGREGATE_MIN_COMPARABLE_GROUPS = 3
_AGGREGATE_MIN_MATCH_FRACTION = 0.7


def _detect_verified_total_categories(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Find category values that are PROVEN (by real arithmetic, not by their
    name) to be a "grand total" row: summing a numeric column over the OTHER
    values in that category, holding other columns fixed, equals the
    candidate value's own sum for most groups.

    This is what correctly tells apart "i_Industry_Total" (a genuine sum of
    business-size bands) from "Total income" (just a metric that happens to
    be named with the word "total" but is NOT the sum of unrelated metrics
    like "Activity unit" in the same column) - the latter simply won't pass
    the arithmetic check.
    """
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() > 0
        and float(df[c].isna().mean()) < _SPARSE_COLUMN_THRESHOLD
    ]
    # Columns usable as GROUP KEYS to hold "everything else" fixed while
    # testing column `col` - any low-cardinality, non-sparse column (numeric
    # like `year`, or text like `variable`), not just text columns.
    group_key_candidates = [
        c for c in df.columns
        if (pd.api.types.is_numeric_dtype(df[c]) or str(df[c].dtype) in ("object", "string", "str"))
        and float(df[c].isna().mean()) < _SPARSE_COLUMN_THRESHOLD
        and 1 < df[c].nunique(dropna=True) <= _MAX_UNIQUE_VALUES_LISTED
    ]
    # Columns to actually TEST for a hidden "grand total" category - text only.
    candidate_cols = [
        c for c in group_key_candidates if str(df[c].dtype) in ("object", "string", "str")
    ]
    if not numeric_cols or len(candidate_cols) < 1 or len(group_key_candidates) < 2:
        return {}

    verified: Dict[str, set] = {}
    for numeric_col in numeric_cols[:5]:
        for col in candidate_cols:
            # Hold EVERY other low-cardinality dimension fixed (not just the
            # two smallest) - otherwise unrelated rows get summed together
            # and can coincidentally "match", producing false positives.
            other_cols = sorted(
                (c for c in group_key_candidates if c != col),
                key=lambda c: df[c].nunique(dropna=True),
            )[:4]
            if not other_cols:
                continue
            try:
                pivot = df.pivot_table(
                    index=other_cols, columns=col, values=numeric_col, aggfunc="sum"
                )
            except Exception:  # noqa: BLE001
                continue
            for candidate_value in pivot.columns:
                others_sum = pivot.drop(columns=[candidate_value]).sum(axis=1, skipna=True)
                candidate_series = pivot[candidate_value]
                comparable = candidate_series.notna() & (others_sum.abs() > 1e-9)
                if int(comparable.sum()) < _AGGREGATE_MIN_COMPARABLE_GROUPS:
                    continue
                ratio = (
                    (candidate_series[comparable] - others_sum[comparable]).abs()
                    / others_sum[comparable].abs()
                )
                if float((ratio < _AGGREGATE_MATCH_TOLERANCE).mean()) > _AGGREGATE_MIN_MATCH_FRACTION:
                    verified.setdefault(col, set()).add(str(candidate_value))

    return {col: sorted(values) for col, values in verified.items()}


def _profile_dataframe(df: pd.DataFrame, verified_totals: Optional[Dict[str, List[str]]] = None) -> str:
    """Build a compact, accurate description of the dataframe for the LLM."""
    verified_totals = verified_totals or {}
    lines = [f"Total rows: {len(df)}", "Columns:"]
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        nunique = int(series.nunique(dropna=True))
        missing_frac = float(series.isna().mean())
        sparse_note = " (mostly empty)" if missing_frac >= _SPARSE_COLUMN_THRESHOLD else ""

        source_col = (
            col[: -len(_NUMERIC_COMPANION_SUFFIX)] if col.endswith(_NUMERIC_COMPANION_SUFFIX) else None
        )
        if source_col is not None and source_col in df.columns:
            lines.append(
                f"- {col} ({dtype}): auto-converted numeric version of '{source_col}' "
                "(non-numeric entries become NaN, automatically skipped by sum/mean/max). "
                f"ALWAYS use '{col}' instead of '{source_col}' for any numeric sum, mean, max, "
                "min, sort, or comparison."
            )
            continue

        if dtype in ("object", "string", "str") and nunique <= _MAX_UNIQUE_VALUES_LISTED:
            values = sorted(str(v) for v in series.dropna().unique().tolist())
            verified_note = ""
            if col in verified_totals:
                verified_note = (
                    f" -- VERIFIED (by checking the real numbers, not the name): {verified_totals[col]} "
                    "is/are already the sum of the other categories in this column. For an overall/"
                    "combined figure, filter to this value instead of summing the other categories "
                    "(summing both would double-count)."
                )
            lines.append(f"- {col} ({dtype}, {nunique} unique values{sparse_note}): {values}{verified_note}")
        else:
            sample = series.dropna().unique()[:5].tolist()
            lines.append(
                f"- {col} ({dtype}, {nunique} unique values{sparse_note}), example values: {sample}"
            )
    return "\n".join(lines)


def generate_query_expression(
    question: str, profile: str, previous_error: Optional[str] = None
) -> str:
    """Ask the LLM to translate the question into a pandas expression."""
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(QueryPlan)

    human_parts = [f"Dataset profile:\n{profile}", f"User question: {question}"]
    if previous_error:
        human_parts.append(
            f"Your previous expression failed with this error, fix it:\n{previous_error}"
        )

    prompt = ChatPromptTemplate.from_messages(
        [("system", QUERY_SYSTEM_PROMPT), ("human", "\n\n".join(human_parts))]
    )
    chain = prompt | structured_llm
    plan: QueryPlan = chain.invoke({})
    return plan.pandas_expression.strip()


class _SafeExpressionValidator(ast.NodeVisitor):
    """
    Walks the expression tree checking every node against the allow-list,
    while tracking lambda-parameter names as locally bound (not free
    identifiers) so `.apply(lambda x: ...)` can be permitted safely.
    """

    def __init__(self, allowed_names: set) -> None:
        self.allowed_names = allowed_names
        self.bound_scopes: List[set] = []

    def _is_bound(self, name: str) -> bool:
        return any(name in scope for scope in self.bound_scopes)

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"Disallowed syntax in generated expression: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            raise ValueError("Disallowed assignment in generated expression")
        if node.id not in self.allowed_names and not self._is_bound(node.id):
            raise ValueError(f"Disallowed identifier: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_") or node.attr in _BLOCKED_ATTR_NAMES:
            raise ValueError(f"Disallowed attribute access: {node.attr}")
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        arg_names = {a.arg for a in node.args.args}
        self.bound_scopes.append(arg_names)
        self.generic_visit(node)
        self.bound_scopes.pop()


def _validate_expression(expr: str) -> None:
    """Raise ValueError if the expression contains anything unsafe."""
    tree = ast.parse(expr, mode="eval")
    allowed_names = {"df", "pd"} | set(_SAFE_BUILTINS)
    _SafeExpressionValidator(allowed_names).visit(tree)


def execute_query_expression(expr: str, df: pd.DataFrame) -> Tuple[Any, Optional[str]]:
    """Safely evaluate a pandas expression against the real dataframe."""
    try:
        _validate_expression(expr)
        code = compile(ast.parse(expr, mode="eval"), "<query>", "eval")
        safe_globals = {"__builtins__": {}, "pd": pd, **_SAFE_BUILTINS}
        result = eval(code, safe_globals, {"df": df})  # noqa: S307
        return result, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _stringify_result(result: Any) -> Optional[str]:
    """Turn a raw pandas/python result into grounded text, or None if empty."""
    if result is None:
        return None

    if isinstance(result, pd.DataFrame):
        if result.empty:
            return None
        total = len(result)
        preview = result.head(_MAX_RESULT_ROWS)
        text = f"Total matching rows: {total}\n\n{preview.to_string(index=False)}"
        if total > _MAX_RESULT_ROWS:
            text += f"\n... ({total - _MAX_RESULT_ROWS} more rows not shown)"
        return text

    if isinstance(result, pd.Series):
        if result.empty:
            return None
        total = len(result)
        preview = result.head(_MAX_RESULT_ROWS)
        text = f"Total matching values: {total}\n\n{preview.to_string()}"
        if total > _MAX_RESULT_ROWS:
            text += f"\n... ({total - _MAX_RESULT_ROWS} more not shown)"
        return text

    text = str(result).strip()
    return text if text else None


ANSWER_SYSTEM_PROMPT = """You are a data analyst answering a question about a dataset.
You will be given the user's question and the EXACT, already-computed result
from the real data. Write a direct, concise answer (1-4 sentences).

Rules:
- Use ONLY the numbers/values given in the computed result. Never add facts,
  numbers, or context that are not present in it.
- Do not say things like "based on the data provided" - just answer directly.
- If the computed result shows zero matches or is empty, say plainly that
  the data does not contain that.
"""


def phrase_answer(question: str, result_text: str) -> str:
    """Turn a raw computed result into a natural-language answer (no new facts)."""
    llm = get_llm(temperature=0.0)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_SYSTEM_PROMPT),
            ("human", "User question: {question}\n\nComputed result:\n{result_text}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "result_text": result_text})


def run_query_agent(question: str, df: pd.DataFrame) -> str:
    """
    Answer a user's question directly and accurately from the real dataframe.

    This is the "Response" the UI always shows - it changes every time the
    user asks a new question, unlike the one-time Statistics/Charts/
    Insights/Report overview.
    """
    if df is None or df.empty:
        return _NOT_FOUND_MESSAGE

    working_df = _augment_numeric_like_columns(df)
    try:
        verified_totals = _detect_verified_total_categories(working_df)
    except Exception:  # noqa: BLE001
        verified_totals = {}
    profile = _profile_dataframe(working_df, verified_totals)

    try:
        expr = generate_query_expression(question, profile)
    except Exception as exc:  # noqa: BLE001
        return f"Sorry, I couldn't process that question right now: {exc}"

    if not expr or expr.strip().upper() == "NOT_FOUND":
        return _NOT_FOUND_MESSAGE

    result, error = execute_query_expression(expr, working_df)

    if error:
        try:
            expr = generate_query_expression(question, profile, previous_error=f"{expr} -> {error}")
        except Exception as exc:  # noqa: BLE001
            return f"Sorry, I couldn't process that question right now: {exc}"

        if not expr or expr.strip().upper() == "NOT_FOUND":
            return _NOT_FOUND_MESSAGE

        result, error = execute_query_expression(expr, working_df)
        if error:
            return "I couldn't compute an accurate answer for that question from the CSV data."

    result_text = _stringify_result(result)
    if result_text is None:
        return "I looked through the CSV data but found no matching results for that question."

    try:
        return phrase_answer(question, result_text)
    except Exception as exc:  # noqa: BLE001
        return f"Found this in the data, but couldn't phrase it (AI error: {exc}):\n\n{result_text}"
