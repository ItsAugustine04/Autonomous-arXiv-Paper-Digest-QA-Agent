"""
Shared state for the arXiv Digest & QA agent.

This is the single object every node reads from and writes to. Using a
TypedDict (rather than passing loose args between functions) makes the
state shape explicit and lets LangGraph merge partial updates from each
node automatically.
"""

from typing import TypedDict, Literal, Optional, Any


class PaperMetadata(TypedDict):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    pdf_url: str
    categories: list[str]
    published: str  # ISO date string


class Chunk(TypedDict):
    chunk_id: str
    text: str
    section: str        # e.g. "Introduction", "Results" — best-effort
    start_char: int


class Briefing(TypedDict):
    title: str
    authors: list[str]
    arxiv_id: str
    published: str
    link: str
    summary: str                 # 1-paragraph plain-English "why it matters"
    problem_statement: str
    method: list[str]            # bullet points
    key_results: list[str]
    limitations: list[str]       # never left empty — see summarize_node
    suggested_questions: list[str]


class QATurn(TypedDict):
    question: str
    answer: str
    retrieved_chunk_ids: list[str]
    grounded: bool                # False if the agent fell back to "not in paper"


class AgentState(TypedDict, total=False):
    # --- input / intent ---
    raw_query: str
    intent: Literal["paper_id", "topic_search"]

    # --- retrieval ---
    candidates: list[PaperMetadata]
    candidate_count: int
    selected_paper: Optional[PaperMetadata]

    # --- parsing ---
    raw_text: str
    parsed_sections: dict[str, str]
    parse_failed: bool
    parse_failure_reason: Optional[str]

    # --- chunking / retrieval store ---
    chunks: list[Chunk]
    vector_store_ref: Any          # e.g. a Chroma collection handle
    embedding_model_name: str

    # --- output ---
    briefing: Optional[Briefing]

    # --- QA loop (persists across turns) ---
    qa_history: list[QATurn]

    # --- control flow / error handling ---
    status: Literal["ok", "needs_disambiguation", "no_results", "failed"]
    error_message: Optional[str]
    