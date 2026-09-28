"""
Node functions for the arXiv Digest & QA agent.

Each node takes the current AgentState, does one job, and returns a dict
of the fields it changed (LangGraph merges this into state). Keeping nodes
single-purpose and side-effect-explicit makes the graph easy to test node
by node without spinning up the whole pipeline.

LLM calls are routed through `call_llm()` in llm.py so this file stays
provider-agnostic (swap Gemini/Groq/Ollama there, nothing here changes).
"""

import os
import re
import tempfile
import uuid
from datetime import datetime

import arxiv          # pip install arxiv
import pymupdf as fitz            # pip install pymupdf
import chromadb

from state import AgentState, PaperMetadata, Chunk, Briefing, QATurn
from llm import call_llm, embed_texts

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


# ---------------------------------------------------------------------------
# 1. Query Understanding
# ---------------------------------------------------------------------------
def query_understanding_node(state: AgentState) -> dict:
    query = state["raw_query"].strip()
    match = ARXIV_ID_RE.search(query)
    if match:
        return {"intent": "paper_id", "raw_query": match.group(1)}
    return {"intent": "topic_search"}


# ---------------------------------------------------------------------------
# 2. arXiv Retrieval
# ---------------------------------------------------------------------------
def arxiv_retrieval_node(state: AgentState) -> dict:
    client = arxiv.Client()

    if state["intent"] == "paper_id":
        search = arxiv.Search(id_list=[state["raw_query"]])
    else:
        search = arxiv.Search(
            query=state["raw_query"],
            max_results=10,
            sort_by=arxiv.SortCriterion.Relevance,
        )

    results = list(client.results(search))
    candidates: list[PaperMetadata] = [
        {
            "arxiv_id": r.get_short_id(),
            "title": r.title,
            "authors": [a.name for a in r.authors],
            "abstract": r.summary,
            "pdf_url": r.pdf_url,
            "categories": r.categories,
            "published": r.published.isoformat(),
        }
        for r in results
    ]

    if not candidates:
        return {
            "candidates": [],
            "candidate_count": 0,
            "status": "no_results",
            "error_message": (
                f"arXiv returned no results for '{state['raw_query']}'. "
                "Try a broader or differently-phrased topic."
            ),
        }

    return {"candidates": candidates, "candidate_count": len(candidates), "status": "ok"}


# ---------------------------------------------------------------------------
# 3. Selection / Ranking (topic search only — paper_id skips straight through)
# ---------------------------------------------------------------------------
def selection_ranking_node(state: AgentState) -> dict:
    candidates = state["candidates"]

    if state["intent"] == "paper_id" or len(candidates) == 1:
        return {"selected_paper": candidates[0], "status": "ok"}

    # Vague topic, many candidates: rank by cosine similarity of query vs.
    # abstract embeddings rather than spending an LLM call on this.
    query_vec = embed_texts([state["raw_query"]])[0]
    abstract_vecs = embed_texts([c["abstract"] for c in candidates])

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb + 1e-8)

    scored = sorted(
        zip(candidates, abstract_vecs),
        key=lambda pair: cosine(query_vec, pair[1]),
        reverse=True,
    )
    ranked = [c for c, _ in scored]

    # More than a couple of plausible top hits with a vague query: don't
    # silently guess — surface the top few and let the caller disambiguate.
    top_score = cosine(query_vec, scored[0][1])
    second_score = cosine(query_vec, scored[1][1]) if len(scored) > 1 else 0
    ambiguous = (top_score - second_score) < 0.03

    if ambiguous:
        return {
            "candidates": ranked[:5],
            "status": "needs_disambiguation",
            "error_message": (
                "Multiple papers look similarly relevant — pick one by arXiv ID, "
                "or narrow the topic."
            ),
        }

    return {"selected_paper": ranked[0], "status": "ok"}


# ---------------------------------------------------------------------------
# 4. Fetch & Parse
# ---------------------------------------------------------------------------
def fetch_parse_node(state: AgentState) -> dict:
    paper = state["selected_paper"]
    try:
        import urllib.request
        import tempfile
        import os

        temp_dir = tempfile.gettempdir()
        pdf_path = os.path.join(temp_dir, f"{paper['arxiv_id'].replace('/', '_')}.pdf")
        
        print(f"Downloading from: {paper['pdf_url']}")
        urllib.request.urlretrieve(paper["pdf_url"], pdf_path)
        print(f"Downloaded to: {pdf_path}")

        doc = fitz.open(pdf_path)
        full_text = "\n".join(page.get_text() for page in doc)
        doc.close()
        print(f"Extracted {len(full_text)} characters")

        # Language detection
        try:
            from langdetect import detect
            lang = detect(full_text[:1000])
            if lang != "en":
                print(f"Warning: paper appears to be in '{lang}' — results may be degraded")
        except:
            pass

    except Exception as e:
        print(f"ERROR DETAILS: {type(e).__name__}: {e}")
        # Fallback to abstract if PDF fails
        abstract = state["selected_paper"]["abstract"]
        if abstract:
            print("PDF failed, falling back to abstract text")
            return {
                "raw_text": abstract,
                "parsed_sections": {"Abstract": abstract},
                "parse_failed": False,
                "status": "ok",
            }
        return {
            "parse_failed": True,
            "parse_failure_reason": f"download/open failed: {e}",
            "status": "failed",
            "error_message": f"Could not fetch or open the PDF: {e}",
        }

    # Cheap sanity check for scanned/image-only PDFs
    if len(full_text.strip()) < 500:
        return {
            "raw_text": full_text,
            "parse_failed": True,
            "parse_failure_reason": "extracted text too short — likely scanned/image PDF",
            "status": "failed",
            "error_message": (
                "This PDF looks like a scanned or image-based document with no "
                "extractable text layer. OCR isn't in scope for this pipeline, "
                "so I can't produce a grounded briefing for it."
            ),
        }

    sections = _split_into_sections(full_text)
    return {
        "raw_text": full_text,
        "parsed_sections": sections,
        "parse_failed": False,
        "status": "ok",
    }

    


def _split_into_sections(text: str) -> dict[str, str]:
    """Best-effort heading split. arXiv PDFs have no consistent structure,
    so this is intentionally forgiving: unmatched text collapses into
    'body' rather than raising."""
    heading_pattern = re.compile(
        r"\n\s*(\d{1,2}\.?\s+)?(Abstract|Introduction|Related Work|Background|"
        r"Method(?:ology)?|Approach|Experiments?|Results?|Discussion|"
        r"Limitations?|Conclusion|References)\s*\n",
        re.IGNORECASE,
    )
    matches = list(heading_pattern.finditer(text))
    if not matches:
        return {"body": text}

    sections = {}
    for i, m in enumerate(matches):
        name = m.group(2).strip().title()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip()
    return sections


# ---------------------------------------------------------------------------
# 5. Chunk & Embed
# ---------------------------------------------------------------------------
def chunk_embed_node(state: AgentState) -> dict:
    text = state["raw_text"]
    # Short paper handling
    if len(text) < 2000:
        print("Short paper detected, using full text as single chunk")
        chunks: list[Chunk] = [{
            "chunk_id": str(uuid.uuid4())[:8],
            "text": text,
            "section": "full",
            "start_char": 0,
        }]
        client = chromadb.Client()
        collection = client.get_or_create_collection(
            name=f"paper_{state['selected_paper']['arxiv_id'].replace('.', '_')}"
        )
        embeddings = embed_texts([text])
        collection.add(
            ids=[chunks[0]["chunk_id"]],
            documents=[text],
            embeddings=embeddings,
        )
        return {"chunks": chunks, "vector_store_ref": collection, "status": "ok"}
    
    chunk_size, overlap = 1000, 300  # chars, not tokens — good enough here

    raw_chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        raw_chunks.append(text[start:end])
        start += chunk_size - overlap

    chunks: list[Chunk] = [
        {
            "chunk_id": str(uuid.uuid4())[:8],
            "text": c,
            "section": _guess_section(c, state["parsed_sections"]),
            "start_char": i * (chunk_size - overlap),
        }
        for i, c in enumerate(raw_chunks)
    ]

    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name=f"paper_{state['selected_paper']['arxiv_id'].replace('.', '_')}"
    )
    embeddings = embed_texts([c["text"] for c in chunks])
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
    )

    return {"chunks": chunks, "vector_store_ref": collection, "status": "ok"}


def _guess_section(chunk_text: str, sections: dict[str, str]) -> str:
    for name, body in sections.items():
        if chunk_text[:100] in body:
            return name
    return "body"


# ---------------------------------------------------------------------------
# 6. Summarize -> Briefing
# ---------------------------------------------------------------------------
BRIEFING_PROMPT = """You are producing an executive briefing for a researcher \
deciding whether to read this paper in full. Base every claim strictly on the \
text provided below — do not use outside knowledge of the paper or authors.

Paper text (may be truncated):
{text}

Return ONLY a JSON object with exactly these keys:
summary (1 paragraph, plain English, "why this paper matters"),
problem_statement (1-2 sentences),
method (list of bullet-point strings),
key_results (list of bullet-point strings),
limitations (list of bullet-point strings — REQUIRED, non-empty; if the paper \
doesn't state limitations explicitly, infer plausible ones from the method \
and say they are inferred),
suggested_questions (3-5 follow-up questions a careful reader might ask).
"""


SECTION_SUMMARY_PROMPT = """Summarize this section of a research paper in 2-3 sentences:

{text}

Summary:"""

BRIEFING_PROMPT = """You are producing an executive briefing for a researcher. \
Based on the section summaries below, produce a structured summary.

Section summaries:
{text}

Return a plain text response with these sections:
WHY IT MATTERS: (1 paragraph)
PROBLEM: (1-2 sentences)
METHOD: (3-5 bullet points starting with -)
KEY RESULTS: (3-5 bullet points starting with -)
LIMITATIONS: (2-3 bullet points starting with -)
SUGGESTED QUESTIONS: (3 questions starting with -)
"""

def summarize_node(state: AgentState) -> dict:
    print("Starting summarize_node, calling LLM...")
    paper = state["selected_paper"]
    full_text = state["raw_text"]

    # Map: if paper is large, summarize in chunks first
    if len(full_text) > 12000:
        print("Large paper detected, using map-reduce summarization...")
        # Split into ~4000 char sections
        section_size = 4000
        sections = [full_text[i:i+section_size] 
                   for i in range(0, min(len(full_text), 24000), section_size)]
        
        section_summaries = []
        for i, section in enumerate(sections):
            print(f"Summarizing section {i+1}/{len(sections)}...")
            summary = call_llm(SECTION_SUMMARY_PROMPT.format(text=section))
            section_summaries.append(summary)
        
        combined = "\n\n".join(section_summaries)
    else:
        combined = full_text

    # Reduce: produce final briefing from combined summaries
    raw = call_llm(BRIEFING_PROMPT.format(text=combined))
    print(f"LLM raw response: {raw[:300]}...")

    # Parse plain text response
    def extract_section(text, marker, next_markers):
        start = text.find(marker)
        if start == -1:
            return ""
        start = text.find(":", start) + 1
        end = len(text)
        for nm in next_markers:
            pos = text.find(nm, start)
            if pos != -1:
                end = min(end, pos)
        return text[start:end].strip()

    def extract_bullets(text, marker, next_markers):
        section = extract_section(text, marker, next_markers)
        lines = [l.strip().lstrip("-").strip() 
                for l in section.split("\n") 
                if l.strip().startswith("-")]
        return lines if lines else [section]

    briefing: Briefing = {
        "title": paper["title"],
        "authors": paper["authors"],
        "arxiv_id": paper["arxiv_id"],
        "published": paper["published"],
        "link": f"https://arxiv.org/abs/{paper['arxiv_id']}",
        "summary": extract_section(raw, "WHY IT MATTERS", 
                                  ["PROBLEM", "METHOD", "KEY RESULTS", 
                                   "LIMITATIONS", "SUGGESTED"]),
        "problem_statement": extract_section(raw, "PROBLEM", 
                                            ["METHOD", "KEY RESULTS", 
                                             "LIMITATIONS", "SUGGESTED"]),
        "method": extract_bullets(raw, "METHOD", 
                                 ["KEY RESULTS", "LIMITATIONS", "SUGGESTED"]),
        "key_results": extract_bullets(raw, "KEY RESULTS", 
                                      ["LIMITATIONS", "SUGGESTED"]),
        "limitations": extract_bullets(raw, "LIMITATIONS", ["SUGGESTED"]) or 
                      ["Not explicitly stated — see full paper"],
        "suggested_questions": extract_bullets(raw, "SUGGESTED QUESTIONS", []),
    }
    return {"briefing": briefing, "status": "ok"}


def _safe_json(raw: str) -> dict:
    import json

    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# 7. QA Loop (called once per user question, not as a single pass)
# ---------------------------------------------------------------------------
QA_PROMPT = """Answer the question using ONLY the excerpts below. If the excerpts \
don't contain the answer, say plainly that the paper doesn't cover this — do not guess.

Excerpts:
{context}

Question: {question}

Answer:
"""

GROUNDING_MIN_SIMILARITY = 0.01  # below this, treat as "not in paper"

def qa_node(state: AgentState, question: str) -> dict:
    collection = state["vector_store_ref"]
    q_vec = embed_texts([question])[0]

    results = collection.query(query_embeddings=[q_vec], n_results=6)
    retrieved_ids = results["ids"][0]
    retrieved_texts = results["documents"][0]

    if retrieved_texts:
        #context = "\n\n---\n\n".join(retrieved_texts)
        unique_texts = list(dict.fromkeys(retrieved_texts))
        context = "\n\n---\n\n".join(unique_texts)
        answer = call_llm(QA_PROMPT.format(context=context, question=question))
        grounded = True
    else:
        answer = "This doesn't appear to be covered in the paper."
        grounded = False

    turn: QATurn = {
        "question": question,
        "answer": answer,
        "retrieved_chunk_ids": retrieved_ids,
        "grounded": grounded,
    }

    history = state.get("qa_history", [])
    return {"qa_history": history + [turn]}
