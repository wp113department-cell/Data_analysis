"""
graph.py
--------
This file holds ALL the LangGraph orchestration code.

LangGraph is responsible for:
    - Defining the shared State
    - Defining Nodes (one per agent)
    - Defining Edges (including a parallel fan-out/fan-in)
    - Compiling and running the graph

LangChain (from agents.py) is only called *inside* nodes to talk to the LLM.
LangGraph itself never talks to the LLM directly - it just decides
"what runs, in what order, with what data".

Flow:

    START
      |
      v
    planner
     /    \\
    v      v
statistics  visualization      <- run in parallel
    \\      /
     v    v
    insight            <- LangGraph automatically waits for BOTH
      |                    branches before running this node (the "merge")
      v
    report
      |
      v
     END
"""

from __future__ import annotations

import tempfile
from typing import Any, Dict, Iterator, List, Optional, Tuple, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agent import agents


class AgentState(TypedDict, total=False):
    """Shared state passed between every node in the graph."""

    user_question: str
    dataframe: pd.DataFrame
    chart_dir: str
    plan: agents.AnalysisPlan
    statistics: Dict[str, Any]
    charts: List[Dict[str, str]]
    insights: str
    report: str
    error: Optional[str]


# ---------------------------------------------------------------------------
# Nodes - each node is a thin wrapper that calls into agents.py
# ---------------------------------------------------------------------------

def planner_node(state: AgentState) -> Dict[str, Any]:
    """Decide what statistics and charts are needed to answer the question."""
    plan = agents.run_planner_agent(state["user_question"], state["dataframe"])
    return {"plan": plan}


def statistics_node(state: AgentState) -> Dict[str, Any]:
    """Compute statistics for the dataset (runs in parallel with visualization_node)."""
    stats = agents.compute_statistics(state["dataframe"], state["plan"])
    return {"statistics": stats}


def visualization_node(state: AgentState) -> Dict[str, Any]:
    """Create charts relevant to the question (runs in parallel with statistics_node)."""
    charts = agents.create_charts(state["dataframe"], state["plan"], state["chart_dir"])
    return {"charts": charts}


def insight_node(state: AgentState) -> Dict[str, Any]:
    """Generate plain-English insights from the statistics (LLM call)."""
    try:
        insights = agents.run_insight_agent(state["user_question"], state["statistics"])
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
        insights = f"_AI insights unavailable: {exc}_"
        return {"insights": insights, "error": str(exc)}
    return {"insights": insights}


def report_node(state: AgentState) -> Dict[str, Any]:
    """Assemble the final markdown report (LLM call for the summary paragraph)."""
    try:
        report = agents.run_report_agent(
            state["user_question"],
            state["statistics"],
            state["insights"],
            state["charts"],
        )
    except Exception as exc:  # noqa: BLE001
        report = f"# Data Analysis Report\n\n_Report generation failed: {exc}_"
        return {"report": report, "error": str(exc)}
    return {"report": report}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the LangGraph state graph."""
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("statistics", statistics_node)
    graph.add_node("visualization", visualization_node)
    graph.add_node("insight", insight_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "planner")

    # Parallel fan-out: both branches start as soon as the planner finishes.
    graph.add_edge("planner", "statistics")
    graph.add_edge("planner", "visualization")

    # Fan-in / merge: LangGraph runs "insight" only once BOTH parents are done.
    graph.add_edge("statistics", "insight")
    graph.add_edge("visualization", "insight")

    graph.add_edge("insight", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_analysis(df: pd.DataFrame, user_question: str) -> AgentState:
    """
    Convenience entry point used by the Streamlit UI.

    Builds the graph, runs it once for the given dataframe/question, and
    returns the final state (statistics, charts, insights, report).
    """
    compiled_graph = build_graph()
    chart_dir = tempfile.mkdtemp(prefix="ai_data_analyst_charts_")

    initial_state: AgentState = {
        "user_question": user_question,
        "dataframe": df,
        "chart_dir": chart_dir,
    }

    final_state = compiled_graph.invoke(initial_state)
    return final_state


def stream_analysis(df: pd.DataFrame, user_question: str) -> Iterator[Tuple[str, AgentState]]:
    """
    Same as run_analysis, but yields (node_name, state_so_far) each time a
    node finishes, so the UI can show live progress through the pipeline.
    The state from the last yield is the complete final state.
    """
    compiled_graph = build_graph()
    chart_dir = tempfile.mkdtemp(prefix="ai_data_analyst_charts_")

    state: AgentState = {
        "user_question": user_question,
        "dataframe": df,
        "chart_dir": chart_dir,
    }

    for update in compiled_graph.stream(state, stream_mode="updates"):
        for node_name, node_output in update.items():
            state.update(node_output or {})
            yield node_name, state
