"""
Wires the nodes into an explicit state graph. QA is deliberately left
outside the compiled graph and called directly (`qa_node`) in a loop from
main.py, since it's a repeated interactive step, not a one-shot pipeline
stage — forcing it into the same graph would mean re-entering the graph
per question for no benefit.
"""

from langgraph.graph import StateGraph, END

from state import AgentState
from nodes import (
    query_understanding_node,
    arxiv_retrieval_node,
    selection_ranking_node,
    fetch_parse_node,
    chunk_embed_node,
    summarize_node,
)


def route_after_retrieval(state: AgentState) -> str:
    return "end" if state["status"] == "no_results" else "select"


def route_after_selection(state: AgentState) -> str:
    return "end" if state["status"] == "needs_disambiguation" else "fetch"


def route_after_fetch(state: AgentState) -> str:
    return "end" if state["status"] == "failed" else "chunk"


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("understand", query_understanding_node)
    g.add_node("retrieve", arxiv_retrieval_node)
    g.add_node("select", selection_ranking_node)
    g.add_node("fetch", fetch_parse_node)
    g.add_node("chunk", chunk_embed_node)
    g.add_node("summarize", summarize_node)

    g.set_entry_point("understand")
    g.add_edge("understand", "retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieval, {"select": "select", "end": END})
    g.add_conditional_edges("select", route_after_selection, {"fetch": "fetch", "end": END})
    g.add_conditional_edges("fetch", route_after_fetch, {"chunk": "chunk", "end": END})
    g.add_edge("chunk", "summarize")
    g.add_edge("summarize", END)

    return g.compile()