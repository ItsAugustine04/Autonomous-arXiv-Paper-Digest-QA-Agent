"""
CLI entry point.

Usage:
    python main.py "recent work on KV-cache compression for LLMs"
    python main.py 2401.12345
    python main.py https://arxiv.org/abs/2401.12345

Flow: run the compiled graph once to get a briefing, print it, then drop
into an interactive QA loop that calls qa_node() per question against the
same in-memory state (vector store + history persist for the session).
"""

import json
import sys

from graph import build_graph
from nodes import qa_node
from state import AgentState


def print_briefing(briefing: dict) -> None:
    print("\n" + "=" * 70)
    print(f"{briefing['title']}")
    print(f"{', '.join(briefing['authors'])}")
    print(f"arXiv:{briefing['arxiv_id']}  ({briefing['published'][:10]})  {briefing['link']}")
    print("=" * 70)
    print(f"\nWhy it matters:\n  {briefing['summary']}")
    print(f"\nProblem:\n  {briefing['problem_statement']}")
    print("\nMethod:")
    for b in briefing["method"]:
        print(f"  - {b}")
    print("\nKey results:")
    for b in briefing["key_results"]:
        print(f"  - {b}")
    print("\nLimitations:")
    for b in briefing["limitations"]:
        print(f"  - {b}")
    print("\nSuggested follow-up questions:")
    for q in briefing["suggested_questions"]:
        print(f"  - {q}")
    print()

'''
def run_qa_loop(state: AgentState) -> None:
    print("Ask questions about the paper (blank line to quit).\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break

        update = qa_node(state, question)
        state["qa_history"] = update["qa_history"]
        turn = state["qa_history"][-1]

        print(turn["answer"])
        if turn["grounded"]:
            print(f"  (grounded in chunks: {', '.join(turn['retrieved_chunk_ids'])})\n")
        else:
            print()
'''
def run_qa_loop(state: AgentState) -> None:
    print("Ask questions about the paper (blank line to quit).\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break

        update = qa_node(state, question)
        state["qa_history"] = update["qa_history"]
        turn = state["qa_history"][-1]

        print(f"\n{turn['answer']}\n")
        if turn["grounded"]:
            print(f"[grounded in chunks: {', '.join(turn['retrieved_chunk_ids'])}]\n")  

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py <topic | arxiv-id | arxiv-url>")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    graph = build_graph()

    final_state = graph.invoke({"raw_query": query, "qa_history": []})

    if final_state.get("status") in ("no_results", "needs_disambiguation", "failed"):
        print(f"\n{final_state.get('error_message', 'Something went wrong.')}\n")
        if final_state.get("status") == "needs_disambiguation":
            print("Candidates:")
            for c in final_state["candidates"]:
                print(f"  - {c['arxiv_id']}: {c['title']}")
        sys.exit(1)

    print_briefing(final_state["briefing"])

    # Optional: dump the briefing as JSON alongside the printed version.
    with open(f"{final_state['selected_paper']['arxiv_id'].replace('/', '_')}_briefing.json", "w") as f:
        json.dump(final_state["briefing"], f, indent=2)

    run_qa_loop(final_state)


if __name__ == "__main__":
    main()