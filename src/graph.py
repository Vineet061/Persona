"""LangGraph pipeline: PDF -> matches -> formatted text -> JSON file.

    parse_bracket -> [deterministic ok?] -+-> format_output -> save_json -> END
                                           |
                                           +-> load_pdf -> extract_matches (LLM)
                                               -> merge_and_label -> format_output -> save_json -> END

`parse_bracket` (bracket_parser.py) reconstructs every match from the draw
sheet's own numbering/bracket arithmetic -- no model call, no guessing, and
(unlike the LLM path) it can't silently drop matches from a long page. It's
the primary path because it's what the target PDF format actually needs.

It only works for a standard single-elimination "Tournament Software"-style
sheet (contiguous match ids, round sizes that halve as 2^(n-1)..1). If a PDF
doesn't fit that shape, `parse_bracket` raises and the graph falls back to
the original per-page LLM extraction (LangChain + your configured provider)
so a differently-formatted sheet still produces output instead of nothing.

`save_json` always builds the JSON payload (schema in json_export.py); it
only writes it to disk when a `json_path` was passed into `run()`.
"""

import json
import time
from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.scoreSheet.bracket_parser import parse_bracket
from src.scoreSheet.formatter import assign_round_label, format_matches
from src.scoreSheet.json_export import build_export
from src.scoreSheet.llm_extract import extract_matches_from_page
from src.scoreSheet.pdf_extract import PageContent, extract_pages
from src.scoreSheet.schema import Match
from src.scoreSheet.excel_bracket_parser import parse_excel_bracket

DETERMINISTIC_PARSERS = (
    ("deterministic", parse_bracket),  # "T- <id>" sheets: results and schedules
    ("deterministic_excel", parse_excel_bracket),  # spreadsheet-style bracket
)

class PipelineState(TypedDict, total=False):
    pdf_path: str
    json_path: Optional[str]
    start_time: float
    pages: List[PageContent]
    matches: List[Match]
    formatted: str
    json_output: dict
    method: str
    fallback_reason: Optional[str]
    tokens_used: int


def try_deterministic(state: PipelineState) -> PipelineState:
    reasons = []
    for method, parser in DETERMINISTIC_PARSERS:
        try:
            return {"matches": parser(state["pdf_path"]), "method": method}
        except Exception as exc:
            reasons.append(f"{method}: {exc}")
    return {"method": "llm_fallback", "fallback_reason": "; ".join(reasons)}


def route_after_deterministic(state: PipelineState) -> str:
    return "load_pdf" if state["method"] == "llm_fallback" else "format_output"



def load_pdf(state: PipelineState) -> PipelineState:
    return {"pages": extract_pages(state["pdf_path"])}


def extract_matches(state: PipelineState) -> PipelineState:
    matches: List[Match] = []
    tokens_used = 0
    for page in state["pages"]:
        result, tokens = extract_matches_from_page(page.page_number, page.text)
        tokens_used += tokens
        for match in result.matches:
            match.round_label = assign_round_label(page.text, match.match_id)
            matches.append(match)
    return {"matches": matches, "tokens_used": tokens_used}


def merge_and_label(state: PipelineState) -> PipelineState:
    seen = {}
    for match in state["matches"]:
        seen[match.match_id] = match  # last extraction for a given id wins
    ordered = sorted(seen.values(), key=lambda m: m.match_id)
    return {"matches": ordered}


def format_output(state: PipelineState) -> PipelineState:
    return {"formatted": format_matches(state["matches"])}


def save_json(state: PipelineState) -> PipelineState:
    processing_time_ms = round((time.perf_counter() - state["start_time"]) * 1000, 2)
    export = build_export(
        state["pdf_path"],
        state["matches"],
        state["method"],
        processing_time_ms=processing_time_ms,
        tokens_used=state.get("tokens_used", 0),
    )
    json_path = state.get("json_path")
    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)
    return {"json_output": export}






def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("try_deterministic", try_deterministic)
    graph.add_node("load_pdf", load_pdf)
    graph.add_node("extract_matches", extract_matches)
    graph.add_node("merge_and_label", merge_and_label)
    graph.add_node("format_output", format_output)
    graph.add_node("save_json", save_json)

    graph.set_entry_point("try_deterministic")
    graph.add_conditional_edges(
        "try_deterministic",
        route_after_deterministic,
        {"format_output": "format_output", "load_pdf": "load_pdf"},
    )
    graph.add_edge("load_pdf", "extract_matches")
    graph.add_edge("extract_matches", "merge_and_label")
    graph.add_edge("merge_and_label", "format_output")
    graph.add_edge("format_output", "save_json")
    graph.add_edge("save_json", END)

    return graph.compile()


def run(pdf_path: str, json_path: Optional[str] = None) -> PipelineState:
    app = build_graph()
    return app.invoke(
        {"pdf_path": pdf_path, "json_path": json_path, "start_time": time.perf_counter()}
    )
