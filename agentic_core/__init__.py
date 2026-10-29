"""
Deep Research Assistant — Agentic Core (Phase 1: LangGraph migration)

Full flow: plan (+ non-text content decisions) -> human-reviewed outline (with
weightage + user-supplied content) -> research/non-text-fetch/write loop ->
per-section validation (factual + copyright + media SFW/copyright) with
corrective retries -> citation numbering -> per-section formatting -> report
assembly (title/date/version/TOC/citations) -> export.

This package is organized as:
  constants.py      - MAX_REVISIONS / COPYRIGHT_OVERLAP_THRESHOLD / DATA_DIR
  config.py          - ResearchConfig, ReportConfig
  services.py        - LLM/search/vision-checker builders + small text helpers
  content.py          - non-text content generators (images, tables, charts, diagrams)
  state.py            - ResearchState and friends (the LangGraph state shape)
  reference_docs.py   - reference-doc extraction + TF-IDF matching (no RAG)
  nodes_plan.py        - plan / review_outline / build_work_queue nodes
  nodes_research.py    - research / non_text_master / write / validate nodes
  nodes_output.py      - citations / formatting / assembly / export / cleanup
  graph.py             - build_graph() wiring + the __main__ demo

Everything below is re-exported here so existing callers can keep doing
`from agentic_core import build_graph, ResearchConfig, ...` unchanged.

Bugs fixed from the original notebook (flagged, not silently changed):
  1. DeepResearchAssistant.__init__ did `self.webSearchManager = WebSearchManager`
     (the class itself, not the passed-in instance).
  2. synthesize_standard_section() had `return final_report` inside its for-loop,
     so only the FIRST sub-topic per theme was ever written.
  3. ReportConfig.get_standard_themes() referenced `report_format` without
     receiving it as a parameter.

Human-in-the-loop: review_outline uses LangGraph's interrupt() + a checkpointer.
The RESUME payload must be JSON-serializable, so a user-uploaded zip of content
files is NOT passed through Command(resume=...) — the caller (Streamlit) saves
and extracts the zip to `<outline_dir>/contents/` BEFORE resuming, and only a
manifest travels through resume:
    Command(resume={
        "outline": [...],
        "user_provided_content": {"cheese_map.png": {"theme": "...", "sub_topic": "...", "type": "internet-image"}},
    })

Non-text content design notes:
  - Chart content is NOT LLM-generated code. Gemma only produces a small JSON
    chart spec (type/labels/values); a fixed, deterministic matplotlib function
    renders it. No generated code is ever executed — this avoids both the
    reliability problem (small models are weak at code-gen) and the security
    problem (executing LLM output is a real attack surface regardless of model).
  - The "master -> child nodes" delegation is implemented as one node calling
    plain Python functions per content type (optionally in parallel with a
    thread pool later) rather than LangGraph's Send() fan-out API — simpler and
    fully verifiable; Send-based parallelism is a reasonable later optimization
    once this sequential version is proven correct.
  - internet-image needs UNSPLASH_ACCESS_KEY (licensed source, not a raw image
    search, to keep licensing unambiguous). ai-generated-image is a stub — wire
    it to Stable Diffusion/FLUX or a hosted image-gen API. data-table fetch is
    deterministic (pandas.read_html), no LLM involved, so no hallucinated
    numbers. The copyright/SFW check on system-generated media uses Gemini
    3.6 Flash (Gemini 1.5 Pro, mentioned in the original spec, is fully
    shut down as of 2026 — calls to it now 404).

Reference documents (no RAG): reference_docs is now the FIRST node, before
plan, so outline generation can reflect what source material actually exists.
Matching a section to its relevant reference doc(s) uses TF-IDF + cosine
similarity (match_reference_docs) — a basic-NLP, embedding-free alternative to
RAG that's appropriate at this scale (a handful of documents, queried once;
no persistent corpus, no index needed). Reference-doc text is deliberately
kept separate from current_context (web search) in a new
current_reference_context field, so the copyright overlap check in
validate_node — which only ever runs against current_context — never flags
the user's own uploaded material. Reference docs get citation entries but are
exempt from the copyright check, per spec. A new `cleanup` node at the end of
the graph deletes raw uploads + extracted text once the report is exported;
only outline.json and the final report persist. Supported input types: text
PDF, scanned PDF (auto-detected per page), .docx, .html, and images with text
(all via OCR/parsing helpers, no RAG pipeline). New dependencies for this
piece: scikit-learn, pdfplumber (existing), PyMuPDF (fitz), pytesseract,
Pillow, python-docx (existing), beautifulsoup4.

Checked against current (2026) LangChain/LangGraph/Gemini APIs but NOT executed
against real models/API keys in this environment — smoke-test before layering
anything else on top.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from langgraph.types import Command, interrupt  # noqa: F401 (re-exported for callers)

from .config import ReportConfig, ResearchConfig
from .constants import COPYRIGHT_OVERLAP_THRESHOLD, DATA_DIR, MAX_CONTENT_REQUESTS_PER_THEME, MAX_REVISIONS
from .content import (
    DIAGRAM_CONFIGS,
    ContentRequest,
    ContentResult,
    ai_generated_image_content,
    internet_image_content,
    make_chart_content,
    make_data_table_content,
    make_diagram_content,
    render_chart,
    validate_diagram,
)
from .graph import build_graph
from .nodes_output import (
    assemble_report_node,
    assign_citations_node,
    cleanup_node,
    export_node,
    make_format_sections_node,
)
from .nodes_plan import (
    add_content_request,
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
from .reference_docs import (
    REFERENCE_EXTRACTORS,
    create_run_workspace,
    match_reference_docs,
    reference_docs_node,
)
from .services import (
    LLM_PROVIDER_GEMINI_FLASH_LITE,
    LLM_PROVIDER_GEMMA_7B,
    GeminiLLM,
    _linkify_urls,
    _max_overlap_ratio,
    build_llm,
    build_search_tool,
    make_gemini_vision_checker,
    noop_vision_checker,
    parse_llm_json,
    validate_model_file,
)
from .state import ResearchState, SectionRecord, SourceRef, WorkItem

__all__ = [
    "Command", "interrupt",
    "ResearchConfig", "ReportConfig",
    "MAX_REVISIONS", "COPYRIGHT_OVERLAP_THRESHOLD", "DATA_DIR", "MAX_CONTENT_REQUESTS_PER_THEME",
    "validate_model_file", "build_llm", "build_search_tool", "make_gemini_vision_checker",
    "noop_vision_checker",
    "GeminiLLM", "LLM_PROVIDER_GEMMA_7B", "LLM_PROVIDER_GEMINI_FLASH_LITE",
    "parse_llm_json", "_max_overlap_ratio", "_linkify_urls",
    "ContentRequest", "ContentResult", "internet_image_content", "ai_generated_image_content",
    "make_data_table_content", "render_chart", "make_chart_content", "make_diagram_content",
    "DIAGRAM_CONFIGS", "validate_diagram",
    "SourceRef", "WorkItem", "SectionRecord", "ResearchState",
    "REFERENCE_EXTRACTORS", "reference_docs_node", "match_reference_docs", "create_run_workspace",
    "make_plan_node", "review_outline_node", "build_work_queue_node", "make_content_requests_node",
    "add_content_request",
    "make_research_node", "route_after_research", "make_non_text_master_node",
    "make_write_node", "make_validate_node", "route_next", "route_after_validate",
    "assign_citations_node", "make_format_sections_node", "assemble_report_node",
    "cleanup_node", "export_node",
    "build_graph",
]
