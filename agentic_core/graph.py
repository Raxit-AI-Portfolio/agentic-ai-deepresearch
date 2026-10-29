"""Graph assembly.

Full flow: reference_docs -> plan (+ non-text content decisions) ->
human-reviewed outline (with weightage + user-supplied content) ->
research/non-text-fetch/write loop -> per-section validation (factual +
copyright + media SFW/copyright) with corrective retries -> citation
numbering -> per-section formatting -> report assembly
(title/date/version/TOC/citations) -> export -> cleanup."""

from __future__ import annotations

import os

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from .config import ReportConfig, ResearchConfig
from .nodes_output import (
    assemble_report_node,
    assign_citations_node,
    cleanup_node,
    export_node,
    make_format_sections_node,
)
from .nodes_plan import (
    build_work_queue_node,
    make_content_requests_node,
    make_plan_node,
    review_outline_node,
)
from .nodes_research import (
    make_non_text_master_node,
    make_research_node,
    make_validate_node,
    make_write_node,
    route_after_research,
    route_after_validate,
    route_next,
)
from .reference_docs import create_run_workspace, reference_docs_node
from .services import build_llm, build_search_tool, make_gemini_vision_checker, noop_vision_checker
from .state import ResearchState


def build_graph(config: ResearchConfig, report_cfg: ReportConfig,
                 llm=None, search=None, vision_checker=None):
    """llm/search/vision_checker are optional overrides — pass fakes/stubs
    here to test the graph's wiring (routing, validation loop, formatting,
    export) without a GPU, a model file, or any API keys. Leave them None to
    build the real ones from config."""
    llm = llm or build_llm(config)
    search = search or build_search_tool(config)
    if vision_checker is None:
        has_gemini_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
        vision_checker = make_gemini_vision_checker(config.vision_model) if has_gemini_key else noop_vision_checker

    graph = StateGraph(ResearchState)
    graph.add_node("reference_docs", reference_docs_node)
    graph.add_node("plan", make_plan_node(llm, report_cfg))
    graph.add_node("content_requests", make_content_requests_node(llm))
    graph.add_node("review_outline", review_outline_node)
    graph.add_node("build_work_queue", build_work_queue_node)
    graph.add_node("research", make_research_node(search))
    graph.add_node("non_text_master", make_non_text_master_node(llm, search))
    graph.add_node("write", make_write_node(llm))
    graph.add_node("validate", make_validate_node(llm, vision_checker))
    graph.add_node("assign_citations", assign_citations_node)
    graph.add_node("format_sections", make_format_sections_node())
    graph.add_node("assemble_report", assemble_report_node)
    graph.add_node("export", export_node)
    graph.add_node("cleanup", cleanup_node)

    graph.add_edge(START, "reference_docs")
    graph.add_edge("reference_docs", "plan")
    graph.add_edge("plan", "content_requests")
    graph.add_edge("content_requests", "review_outline")
    graph.add_edge("review_outline", "build_work_queue")
    graph.add_conditional_edges("build_work_queue", route_next,
                                 {"research": "research", "write": "write", "finalize": "assign_citations"})
    graph.add_conditional_edges("research", route_after_research,
                                 {"non_text_master": "non_text_master", "write": "write"})
    graph.add_edge("non_text_master", "write")
    graph.add_edge("write", "validate")
    graph.add_conditional_edges("validate", route_after_validate,
                                 {"research": "research", "write": "write",
                                  "non_text_master": "non_text_master", "finalize": "assign_citations"})
    graph.add_edge("assign_citations", "format_sections")
    graph.add_edge("format_sections", "assemble_report")
    graph.add_edge("assemble_report", "export")
    graph.add_edge("export", "cleanup")
    graph.add_edge("cleanup", END)

    return graph.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    import json

    app = build_graph(ResearchConfig(), ReportConfig())
    thread = {"configurable": {"thread_id": "demo-1"}}

    # In Streamlit: save the user's uploaded reference-docs zip to a temp path
    # first, then pass it here — extraction happens before the graph starts.
    workspace = create_run_workspace(reference_zip_path=None)  # e.g. "/tmp/uploaded_refs.zip"

    initial_input = {
        **workspace,
        "topic": "Cheese manufacturing in Italy",
        "report_format": "detailed",
        "output_format": "markdown",
        "source_whitelist": [],
        "source_blacklist": ["pinterest.com"],
    }
    result = app.invoke(initial_input, thread)

    if "__interrupt__" in result:
        draft_outline = result["__interrupt__"][0].value["draft_outline"]
        print("Review this outline (edit sub_topics/queries/content_requests/weightage):")
        print(json.dumps(draft_outline, indent=2))
        # In Streamlit: render as an editable form, save+extract any uploaded
        # zip to <outline_dir>/contents/, then resume with the manifest:
        result = app.invoke(Command(resume={"outline": draft_outline, "user_provided_content": {}}), thread)

    print(result["final_report"])
    if result.get("export_path"):
        print(f"\nExported to: {result['export_path']}")
