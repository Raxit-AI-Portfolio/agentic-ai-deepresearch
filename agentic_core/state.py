"""Graph state — the TypedDicts LangGraph threads through every node."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict

from .content import ContentRequest, ContentResult


class SourceRef(TypedDict, total=False):
    title: str
    url: str
    kind: Literal["web", "reference"]


class WorkItem(TypedDict, total=False):
    kind: Literal["meta", "standard"]
    theme: str
    sub_topic: Optional[str]
    queries: Optional[List[str]]
    weightage: Optional[float]
    content_requests: List[ContentRequest]
    revision_note: Optional[str]
    revision_kind: Optional[Literal[
        "copyright", "factual", "media", "title_splice", "placeholder", "fabrication", "citation",
    ]]
    revision_count: int


class PointerRecord(TypedDict):
    text: str
    citation: Optional[int]


class SectionRecord(TypedDict):
    kind: Literal["meta", "standard"]
    theme: str
    sub_topic: Optional[str]
    content: str
    weightage: Optional[float]
    sources: List[SourceRef]
    media: List[ContentResult]
    status: Literal["pending", "needs_revision", "approved", "approved_with_warning"]
    validation_notes: Optional[str]
    revision_count: int


class ResearchState(TypedDict, total=False):
    # invocation inputs
    topic: str
    report_format: str
    output_format: Literal["json", "xml", "markdown", "html", "pdf", "docx"]
    source_whitelist: List[str]
    source_blacklist: List[str]
    # planning / persistence
    run_id: str
    outline_dir: str
    reference_docs_path: Optional[str]
    reference_docs: Dict[str, dict]
    themes: List[str]
    standard_themes: List[str]
    draft_outline: List[dict]
    reviewed_outline: List[dict]
    user_provided_content: Dict[str, dict]
    # execution loop
    work_queue: List[WorkItem]
    current_context: str
    current_reference_context: str
    current_sources: List[SourceRef]
    current_media: List[ContentResult]
    current_section_draft: Optional[SectionRecord]
    current_pointers: List[PointerRecord]
    last_theme_written: Optional[str]
    seen_urls: List[str]
    sections: List[SectionRecord]
    used_pointers: List[PointerRecord]
    # output
    citation_map: Dict[str, int]
    citations: List[SourceRef]
    formatted_sections: List[dict]
    final_report: str
    export_path: Optional[str]
