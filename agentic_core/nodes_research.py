"""Nodes — make_research_node, make_non_text_master_node, make_write_node, make_validate_node

Reference-doc text is deliberately kept separate from current_context (web
search) in current_reference_context, so the copyright overlap check in
validate_node — which only ever runs against current_context — never flags
the user's own uploaded material.

The "master -> child nodes" delegation (non_text_master_node) is implemented
as one node calling plain Python functions per content type (optionally in
parallel with a thread pool later) rather than LangGraph's Send() fan-out
API — simpler and fully verifiable; Send-based parallelism is a reasonable
later optimization once this sequential version is proven correct."""

from __future__ import annotations

import html
import re
from datetime import date
from typing import List, Literal
from urllib.parse import urlparse

from langchain_community.llms import LlamaCpp
from langchain_community.utilities import GoogleSerperAPIWrapper

from .constants import COPYRIGHT_OVERLAP_THRESHOLD, MAX_REVISIONS
from .content import (
    ContentResult,
    ai_generated_image_content,
    internet_image_content,
    make_chart_content,
    make_data_table_content,
    make_diagram_content,
)
from .reference_docs import match_reference_docs
from .services import (
    _max_overlap_ratio,
    drop_duplicate_sentences,
    drop_self_duplicate_sentences,
    filter_novel_pointers,
    filter_relevant_results,
    fix_missing_spaces,
    split_sentences,
    strip_llm_preamble,
)
from .state import ResearchState, SectionRecord, WorkItem


# Per-theme purpose guidance for meta sections (make_write_node). Without this,
# every meta theme got the exact same generic "high-level narrative synthesis"
# instruction, which — combined with a quantized 7B model's tendency to reach
# for the most familiar framing — led Overview/Next Steps/Conclusion to all
# open with near-identical scene-setting sentences instead of serving their
# distinct purposes. Covers every theme in ReportConfig.get_meta_themes().
_META_THEME_GUIDANCE = {
    "EXECUTIVE SUMMARY": "Condense the report's most decision-relevant findings and recommendations into a "
                          "standalone summary a busy reader could act on without reading the rest of the report.",
    "ABSTRACT": "Concisely state the report's scope, approach, and single most important finding, in the "
                "style of an academic abstract.",
    "INTRODUCTION": "State the report's purpose, scope, and structure — what question it answers and how it's "
                     "organized. Do not summarize findings yet; that comes later.",
    "OVERVIEW": "Set context and scope for the topic at a high level, orienting the reader before the "
                "detailed sections that follow.",
    "CONCLUSION": "Synthesize the overall implications and significance of the findings — what they mean for "
                   "the reader, not a restatement of the facts themselves.",
    "NEXT STEPS": "Recommend concrete, forward-looking actions or areas for further investigation based on "
                   "the report's findings. Do not restate what has already been found.",
}
_DEFAULT_META_GUIDANCE = "Focus entirely on high-level narrative synthesis and context."

_TOPIC_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
# "( )%" / "(%)" — a percentage the model started to write, had no real
# figure for, and left as a literal empty-parens placeholder instead of
# dropping the claim or citing a real number (open-issues.md).
_PLACEHOLDER_RE = re.compile(r"\(\s*\)\s*%?|\(\s*%\s*\)")
# "23rd quarter" — a confident-sounding but nonsensical fabricated detail;
# quarters only run 1-4 (open-issues.md).
_INVALID_QUARTER_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\s+quarter\b", re.IGNORECASE)
# "...in the year of [9]" / "...in [20]" where the model substituted a bare
# citation bracket for a year it didn't have (open-issues.md).
_MALFORMED_YEAR_CITATION_RE = re.compile(
    r"\b(?:in\s+the\s+year|the\s+year\s+of|year\s+of|the\s+year)\s*\[\d+\]", re.IGNORECASE
)
# Groups research_node's "[N] Title: ...\n[N] URL: ...\n[N] Summary: ..."
# context lines back into one blob per citation number, for the
# miscitation check below.
_CONTEXT_BLOCK_RE = re.compile(
    r"\[(\d+)\] Title: (.*?)\n\[\1\] URL: .*?\n\[\1\] Summary: (.*?)(?=\n\[\d+\] Title:|\Z)", re.DOTALL
)
# A rough, dependency-free named-entity proxy: 2+ consecutive Title-Case
# words (catches person/organization/place names like "Steven Frank" or
# "European Union") without pulling in an NER library — appropriate at this
# scale (a handful of short pointer phrases, checked once). Deliberately
# requires 2+ words: a single capitalized word is indistinguishable from
# ordinary sentence-initial capitalization and would be far too noisy.
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")


def _date_grounding_clause(topic: str) -> str:
    """The 'don't invent a year' instruction, shared by every generation
    prompt (write_node's meta/standard branches, pointer extraction,
    _extend_content). Deliberately short — the real enforcement for
    standard sections is now _valid_pointer_year below, which drops a
    pointer outright rather than trusting the model to interpret a longer
    instruction correctly (a longer version of this clause was tried first
    and worked, but made every prompt noticeably longer for something code
    can check for free). Naming the topic's year explicitly when one is
    findable still costs almost nothing and measurably helps."""
    today = date.today().isoformat()
    match = _TOPIC_YEAR_RE.search(topic)
    if match:
        return f"Today is {today}. This report is about {match.group(0)} — never use a different year."
    return f"Today is {today}. Use only the year(s) named in the topic or Research Notes — never invent one."


def _valid_pointer_year(pointer_text: str, topic: str, context: str) -> bool:
    """A pointer mentioning a year is kept only if that year is the report's
    own topic year — or, when the topic names no year, if it's at least
    genuinely present in the Research Notes it was extracted from. Closes
    the exact loophole that caused the year-hallucination bug: a pointer
    could carry a year that's real (genuinely present in the Research
    Notes) but wrong for THIS report — e.g. a source's historical-
    comparison year ("up from 12% in 2015") mistaken for the report's own
    timeframe, observed directly as a 2025-topic report narrating its
    findings as happening "in 2015." A pointer mentioning no year at all is
    always fine — this only rejects a pointer that names a WRONG year, not
    one that's merely undated."""
    years_in_pointer = {m.group(0) for m in _TOPIC_YEAR_RE.finditer(pointer_text)}
    if not years_in_pointer:
        return True
    topic_match = _TOPIC_YEAR_RE.search(topic)
    if topic_match:
        return years_in_pointer.issubset({topic_match.group(0)})
    context_years = {m.group(0) for m in _TOPIC_YEAR_RE.finditer(context)}
    return years_in_pointer.issubset(context_years)


def _ungrounded_entities(pointer_text: str, context: str) -> List[str]:
    """Named entities (see _ENTITY_RE) mentioned in a pointer that don't
    appear anywhere in the Research Notes it was supposedly extracted
    from — a specific person/organization/place name invented rather than
    drawn from source material is exactly the shape of hallucination a
    generic factual-grounding check tends to miss (it reads as a very
    specific, confident-sounding claim). Case-insensitive substring check,
    consistent with how the rest of this module treats "is this actually in
    the source text" questions (_max_overlap_ratio, filter_relevant_results)."""
    context_lower = context.lower()
    return [e for e in _ENTITY_RE.findall(pointer_text) if e.lower() not in context_lower]


def _spliced_source_titles(content: str, sources: List[dict], min_len: int = 15, ratio: float = 0.5) -> List[str]:
    """Flags a source TITLE copied near-verbatim into the drafted prose
    instead of being paraphrased and cited as '[N]' — observed twice (see
    open-issues.md): research_node's context lines literally read '[N]
    Title: {title}', and the model sometimes copies that title text into
    its own sentence rather than treating it purely as reference metadata.
    Short/generic titles (under min_len chars) are skipped — too likely to
    overlap a drafted sentence by coincidence rather than actual copying."""
    hits = []
    for src in sources or []:
        title = (src.get("title") or "").strip()
        if len(title) < min_len:
            continue
        if _max_overlap_ratio(title, content) >= ratio:
            hits.append(title)
    return hits


def _invalid_quarter_claims(content: str) -> List[str]:
    """Flags a fabricated 'Nth quarter' claim where N isn't 1-4 (e.g. '23rd
    quarter') — same class of confident-but-nonsensical fabricated detail as
    the year-hallucination bug, different phrase/mechanism (open-issues.md)."""
    return [m.group(0) for m in _INVALID_QUARTER_RE.finditer(content) if int(m.group(1)) not in (1, 2, 3, 4)]


def _context_blobs(context: str) -> dict:
    """Groups research_node's per-citation context lines into one
    title+summary blob per citation number, keyed by that number, for
    _find_miscited_sentences below."""
    return {int(n): f"{title} {summary}" for n, title, summary in _CONTEXT_BLOCK_RE.findall(context or "")}


def _find_invalid_citation_numbers(content: str, valid_numbers: set) -> List[int]:
    """Flags any bare '[N]' in the FINAL drafted prose that isn't one of the
    real citation numbers assigned to a source in this item's Research Notes.
    write_node's pointer list is already checked this way, but the free-form
    paragraph written around those pointers reads the full raw Research
    Notes directly and can still fabricate an out-of-range number (observed:
    a nonexistent '[20]' in a 13-citation report — open-issues.md)."""
    found = {int(n) for n in re.findall(r"\[(\d+)\]", content)}
    return sorted(n for n in found if n not in valid_numbers)


def _find_miscited_sentences(content: str, context_blobs: dict, margin: float = 0.15) -> List[str]:
    """For each sentence citing exactly one '[N]', checks whether that
    sentence's text actually reads as closer (TF-IDF + cosine, same
    embedding-free approach used elsewhere in this module) to a DIFFERENT
    source's title+summary than to the one it cites — the shape of the
    observed miscitation bug: content paraphrased from one source but
    attributed to an unrelated source's number (open-issues.md). A
    heuristic, not a proof — only flags a clear, wide margin, and only when
    there are at least two real sources to tell apart."""
    if len(context_blobs) < 2:
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    nums = sorted(context_blobs)
    corpus = [context_blobs[n] for n in nums]
    hits = []
    for sentence in split_sentences(content):
        cited = [int(c) for c in re.findall(r"\[(\d+)\]", sentence)]
        if len(cited) != 1 or cited[0] not in context_blobs:
            continue
        try:
            matrix = TfidfVectorizer(stop_words="english").fit_transform(corpus + [sentence])
        except ValueError:
            continue
        sims = cosine_similarity(matrix[-1:], matrix[:-1])[0]
        cited_idx = nums.index(cited[0])
        best_idx = int(sims.argmax())
        if best_idx != cited_idx and sims[best_idx] - sims[cited_idx] > margin and sims[best_idx] > 0.1:
            hits.append(f"cites [{cited[0]}] but reads closer to [{nums[best_idx]}]: {sentence!r}")
    return hits


def _correct_wrong_years(text: str, topic: str) -> str:
    """Deterministically corrects a sentence that names a year OTHER than
    the report's own topic year, when that's the ONLY year the sentence
    mentions — the exact failure pattern observed repeatedly (e.g. an
    entire 2025-topic section narrated as "The year 2015 was..."), in both
    a meta section (no pointers to validate at all) and a standard
    section's final prose (which reads the FULL raw Research Notes
    directly, not just the year-checked pointer list — _valid_pointer_year
    only guards pointers, not the free-form paragraph written around them).
    A sentence that ALSO correctly mentions the topic's own year is left
    completely untouched, even if it names another year too (a genuine
    "grown from 12% in 2020 to 19% in 2025" comparison) — only a sentence
    with NO reference to the correct year at all gets its lone wrong year
    swapped in. Prompt instructions alone weren't reliable here even after
    being tightened twice (see open-issues.md history); this fixes it in
    code instead, the same way duplication is."""
    topic_match = _TOPIC_YEAR_RE.search(topic)
    if not topic_match:
        return text
    topic_year = topic_match.group(0)
    fixed = []
    for s in split_sentences(text):
        years = {m.group(0) for m in _TOPIC_YEAR_RE.finditer(s)}
        if topic_year in years or len(years) != 1:
            fixed.append(s)
            continue
        wrong_year = next(iter(years))
        fixed.append(re.sub(rf"\b{wrong_year}\b", topic_year, s))
    return " ".join(fixed)


def _parse_pointer_lines(text: str) -> List[dict]:
    """Parses '<pointer text> | [N]' lines from the pointer-extraction call
    in make_write_node — same flat pipe-delimited format as
    nodes_plan.py::_parse_sub_topic_lines, for the same reason: a small
    quantized model reproduces one line of plain text per item far more
    reliably than any structured (JSON, numbered-list) format."""
    pointers = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip().strip("`")
        if not line or "|" not in line:
            continue
        pointer_part, _, citation_part = line.partition("|")
        pointer_text = re.sub(r"<[^>]*>", " ", pointer_part).strip(" \"'*-")
        pointer_text = re.sub(r"^[\s\-*\d.)]+", "", pointer_text).strip()
        if not pointer_text or not re.search(r"[A-Za-z]", pointer_text):
            continue
        match = re.search(r"\d+", citation_part)
        pointers.append({"text": pointer_text, "citation": int(match.group(0)) if match else None})
    return pointers


def _extend_content(llm: LlamaCpp, state: ResearchState, item: WorkItem, kept_text: str) -> str:
    """One narrow follow-up call, used only by validate_node when removing
    duplicate sentences (drop_duplicate_sentences) leaves a section too
    thin. Asks specifically for a couple of NEW sentences rather than
    re-running the full write prompt and hoping a blind retry comes out
    different — which repeatedly didn't (see GRAPH.md)."""
    label = item.get("sub_topic") or item["theme"]
    existing_block = kept_text or "(nothing — write fresh content)"
    date_clause = _date_grounding_clause(state['topic'])
    # Output ONLY the sentences — no lead-in. A small model asked to "write
    # 2-3 sentences" sometimes prefaces its answer with something like
    # 'Sure, here are the new sentences about "X":' before the actual
    # content — observed leaking verbatim into a published report. The
    # instruction alone isn't trusted to prevent this (see the "don't
    # repeat" instructions that didn't work either); strip_llm_preamble
    # below is the actual enforcement.
    no_preamble_clause = ('Output ONLY the new sentences themselves — no introduction, no lead-in like "Sure, '
                           'here are..." or "Here is...", no restating the section title.')
    if item["kind"] == "meta":
        theme_guidance = _META_THEME_GUIDANCE.get(item["theme"].strip().upper(), _DEFAULT_META_GUIDANCE)
        prompt = f"""<start_of_turn>user
"{item['theme']}" section of a report on "{state['topic']}". {theme_guidance}
Write 2-3 new sentences continuing this section — do not repeat anything below. {date_clause} {no_preamble_clause}
Already written:
{existing_block}
<end_of_turn>
<start_of_turn>model
"""
    else:
        ref_block = f"\n\nUser-Provided Reference Material:\n{state['current_reference_context']}" \
            if state.get("current_reference_context") else ""
        prompt = f"""<start_of_turn>user
Research Notes:
{state.get('current_context', '')}{ref_block}

Write 2-3 new sentences about "{label}" using only facts from the Research Notes above not already covered below. {date_clause} {no_preamble_clause}
Already written (do not repeat):
{existing_block}
<end_of_turn>
<start_of_turn>model
"""
    addition = llm.invoke(prompt, max_tokens=150).strip()
    addition = strip_llm_preamble(addition)
    addition = fix_missing_spaces(addition)
    return re.sub(r"[ \t]{2,}", " ", addition)


# ==========================================
# NODES
# ==========================================

def make_research_node(search: GoogleSerperAPIWrapper):
    def research_node(state: ResearchState) -> dict:
        """Fetches web search results for the current work item, applies source
        whitelist/blacklist, and returns a context string + list of sources for
        the write_node. Also matches reference docs (TF-IDF) and returns their text in current_reference_context, but this is deliberately kept separate from
        current_context so the copyright overlap check in validate_node never flags user-supplied reference material.

        Each source is assigned its GLOBAL citation number here, the first
        time it's seen across the whole run — not a per-section-local index —
        so the number shown to write_node (and that it's told to cite inline
        as "[N]") is the same number that ends up in the final Citations
        list. citation_map/citations accumulate across every research_node
        call; assign_citations_node just reads the final accumulated result.
        """
        item = state["work_queue"][0]
        whitelist = [d.lower() for d in (state.get("source_whitelist") or [])]
        blacklist = [d.lower() for d in (state.get("source_blacklist") or [])]
        seen_urls = list(state.get("seen_urls", []))
        citation_map = dict(state.get("citation_map", {}))
        citations = list(state.get("citations", []))

        def cite(source) -> int:
            url = source["url"]
            if url not in citation_map:
                citation_map[url] = len(citations) + 1
                citations.append(source)
            return citation_map[url]

        queries = list(item.get("queries") or [])
        if item.get("revision_note") and item.get("revision_kind") == "factual":
            queries = queries + [f"{state['topic']} {item.get('sub_topic', '')} {item['revision_note']}"[:200]]

        results = []
        for query in queries:
            q = query
            if whitelist:
                q += " (" + " OR ".join(f"site:{d}" for d in whitelist) + ")"
            if blacklist:
                q += " " + " ".join(f"-site:{d}" for d in blacklist)
            data = search.results(q)
            # Serper sometimes returns results with almost no real
            # connection to the query (an unrelated company's annual
            # report, a decades-old PDF) that the model then dutifully but
            # nonsensically cites — observed directly in a German
            # car-market report. Filtered by relevance to THIS query
            # specifically (TF-IDF + cosine, see filter_relevant_results)
            # before anything else runs on it.
            organic = filter_relevant_results(query, data.get("organic", []))
            for res in organic:
                url = res.get("link", "")
                if not url or url in seen_urls:
                    continue
                domain = urlparse(url).netloc.lower()
                if blacklist and any(bad in domain for bad in blacklist):
                    continue
                if whitelist and not any(good in domain for good in whitelist):
                    continue
                seen_urls.append(url)
                # Serper's title/snippet text sometimes carries raw HTML
                # entities (e.g. "&amp;") that the model then copies
                # verbatim into generated prose — observed directly as a
                # stray "&amp;" in a published report. Decoded once here,
                # at the source, rather than trying to catch it later.
                results.append({
                    "title": html.unescape(res.get("title", "")),
                    "url": url,
                    "snippet": html.unescape(res.get("snippet", "")),
                })

        sources = []
        context_lines = []
        for r in results[:20]:
            source = {"title": r["title"], "url": r["url"], "kind": "web"}
            num = cite(source)
            sources.append(source)
            context_lines.append(
                f"[{num}] Title: {r['title']}\n[{num}] URL: {r['url']}\n[{num}] Summary: {r['snippet']}"
            )

        # Reference-doc matching (TF-IDF, no RAG — see match_reference_docs).
        # This text is kept OUT of current_context deliberately: the copyright
        # overlap check in validate_node only ever runs against current_context,
        # so user-supplied reference material is never flagged for "copying".
        reference_context = ""
        ref_docs = state.get("reference_docs", {})
        if ref_docs and item["kind"] == "standard":
            matched = match_reference_docs(item.get("queries"), item.get("sub_topic", ""), ref_docs)
            ref_texts = []
            for fname in matched:
                with open(ref_docs[fname]["text_path"], "r", encoding="utf-8") as f:
                    text = f.read()[:3000]
                source = {"title": fname, "url": f"reference:{fname}", "kind": "reference"}
                num = cite(source)
                ref_texts.append(f"Reference Document [{num}]: {fname}\n{text}")
                sources.append(source)
            reference_context = "\n\n".join(ref_texts)

        return {
            "current_context": "\n".join(context_lines),
            "current_reference_context": reference_context,
            "current_sources": sources,
            "seen_urls": seen_urls,
            "citation_map": citation_map,
            "citations": citations,
        }

    # Output from this node goes route_after_research() to decide route.
    return research_node

def make_non_text_master_node(llm: LlamaCpp, search: GoogleSerperAPIWrapper):
    """Delegates to the content generators (the 'child nodes') as plain
    function calls and collects results — see module docstring for why this
    is sequential Python calls rather than LangGraph's Send() fan-out."""
    generators = {
        "internet-image": lambda desc, _t: internet_image_content(desc),
        "ai-generated-image": lambda desc, _t: ai_generated_image_content(desc),
        "data-table": lambda desc, _t: make_data_table_content(search)(desc),
        "chart": make_chart_content(llm),
        # diagram_type is only relevant here — content_requests_node's
        # automatic decisions never set it (always defaults to "flowchart"),
        # but a manually-added request (add_content_request(..., diagram_type=
        # "mindmap")) can pick any of DIAGRAM_CONFIGS' other types.
        "diagram": lambda desc, diagram_type: make_diagram_content(llm)(desc, diagram_type or "flowchart"),
    }

    def non_text_master_node(state: ResearchState) -> dict:
        item = state["work_queue"][0]
        media: List[ContentResult] = []
        for req in item.get("content_requests", []) or []:
            if req.get("user_supplied_path"):
                # data-table/diagram render as markup (table/mermaid text),
                # not a file path — read the supplied file's text content
                # instead of treating it as an image.
                if req["type"] in ("data-table", "diagram"):
                    with open(req["user_supplied_path"], "r", encoding="utf-8") as f:
                        markup = f.read()
                    media.append({"type": req["type"], "markup": markup,
                                   "caption": req["description"], "source": "user_supplied", "status": "approved"})
                else:
                    media.append({"type": req["type"], "path": req["user_supplied_path"],
                                   "caption": req["description"], "source": "user_supplied", "status": "approved"})
                continue
            fn = generators.get(req["type"])
            if fn:
                second_arg = req.get("diagram_type") if req["type"] == "diagram" else state.get("_table_data_hint")
                result = fn(req["description"], second_arg)
                if not result.get("path") and not result.get("markup"):
                    print(f"non_text_master_node: {req['type']} for {item.get('sub_topic')!r} "
                          f"produced nothing — {result.get('validation_notes')}")
                elif result.get("status") == "rejected":
                    # A rejected item (e.g. make_diagram_content's Mermaid
                    # validity check) still has "markup" set to whatever the
                    # model actually wrote — the "produced nothing" check
                    # above never fires for it, which left this failure mode
                    # completely invisible. Print the raw output so it's
                    # possible to see WHY it was rejected, not just that it
                    # was — this is what turns into a TODO comment in the
                    # final report instead of embedding broken content.
                    print(f"non_text_master_node: {req['type']} for {item.get('sub_topic')!r} "
                          f"was rejected — {result.get('validation_notes')}\nRaw output was:\n{result.get('markup')!r}\n")
                media.append(result)
        return {"current_media": media}

    # Output from this node goes to the write_node.
    return non_text_master_node


def make_write_node(llm: LlamaCpp):
    def write_node(state: ResearchState) -> dict:
        """Writes the current work item to a draft section, using the current
        context (web search) and current_reference_context (user-supplied reference docs) as the source material. 
        The draft is then passed to validate_node for copyright/factual checks and, 
        if applicable, vision checks on any SYSTEM-GENERATED media."""
        
        item = state["work_queue"][0]
        topic = state["topic"]
        revision_clause = ""

        # A "media" revision_kind (see validate_node/route_after_validate)
        # means only the diagram/table/photo failed and needs a second
        # generation attempt via non_text_master_node — the TEXT was already
        # fine, so it must NOT be rewritten around the media failure reason
        # (e.g. a Mermaid validation error) as if it were a content problem.
        # non_text_master_node always flows into write_node regardless of
        # why it ran (see graph.py's fixed edge), so this node is the one
        # place that has to make that distinction.
        if item.get("revision_note") and item.get("revision_kind") != "media":
            revision_clause = (
                f"\nRevision required: {item['revision_note']}\n"
                "Rewrite the section to address this before anything else.\n"
            )

        novel_pointers: List[dict] = []

        if item["kind"] == "meta":
            max_tokens = 350
            # Meta sections synthesize from what's already written, not
            # their own research, so the pointer pipeline below doesn't
            # apply to them. They used to also get a "here's what's already
            # written, don't restate it" clause, but that instruction wasn't
            # reliably followed (see GRAPH.md) — duplication is now caught
            # and fixed deterministically in validate_node instead, so the
            # prompt itself only needs to state this section's own purpose.
            date_clause = "\n" + _date_grounding_clause(topic)
            theme_guidance = _META_THEME_GUIDANCE.get(item["theme"].strip().upper(), _DEFAULT_META_GUIDANCE)
            prompt = f"""<start_of_turn>user
Write one paragraph for the "{item['theme']}" section of a report on "{topic}". {theme_guidance} No bullet points. Output ONLY the paragraph itself — no lead-in like "Sure, here is..." or "Here's the paragraph:", no restating the section title before the paragraph.{date_clause}{revision_clause}
<end_of_turn>
<start_of_turn>model
"""
        else:
            max_tokens = int(200 + (item.get("weightage") or 0) * 4)
            ref_block = ""
            if state.get("current_reference_context"):
                ref_block = f"\n\nUser-Provided Reference Material:\n{state['current_reference_context']}"
            date_clause = "\n" + _date_grounding_clause(topic)
            # A sub-topic that ended up with no real search queries (e.g. the
            # model echoed the theme name back with nothing after it — see
            # _parse_sub_topic_lines) reaches here with current_context blank.
            # The citation instruction was previously unconditional, and the
            # model would still hallucinate bracketed numbers like "[1]"/"[4]"
            # with nothing to actually cite. Since citation_map accumulates
            # globally across the whole run, those fabricated numbers could
            # later validate as "real" once other sections' research
            # populated the map — misattributing citations to sources that
            # sections never referenced. Only ask for citations when there's
            # something to cite from.
            # The citation-format example used to be a real-looking sentence
            # ("...reached 680 million kilograms [3]."), meant only to show
            # HOW to format a citation — but the model echoed that exact
            # fabricated fact into unrelated reports (observed verbatim in
            # both a car-market report and a cancer-treatment report, output
            # nonsensically as "680 million kilograms of treated tissue").
            # It wasn't reading the number as an example to discard, it was
            # copying it as a usable fact. Using an abstract <fact>/[N]
            # template instead of a real sentence removes anything
            # fact-shaped for it to copy in the first place.
            has_context = bool(state.get("current_context")) or bool(state.get("current_reference_context"))
            citation_clause = (
                '\nCite each fact inline as a bare bracketed number placed right after the sentence stating it, '
                'in the literal template form "<fact from the Research Notes> [N]" — the number goes directly '
                'after the fact, with no other words or punctuation around it. Never write "Source" or '
                '"(Source N)". Only use numbers that already appear in the Research Notes below — never invent '
                "one, and never reuse any wording or figures from this instruction itself; every number and "
                "fact in your paragraph must come from the Research Notes, not from this example template."
                if has_context else
                "\nNo Research Notes are available — write from general knowledge and do not include bracketed "
                "citation numbers."
            )

            # Fact-level novelty enforcement. Telling the model NOT to repeat
            # other sections (a negative instruction) failed twice: first it
            # ignored the instruction outright, then — when handed the prior
            # section's raw text framed as something to "continue from" — it
            # literally copied that text and appended to it. Neither relies
            # on anything a small quantized model reliably does.
            #
            # Instead: extract short fact pointers from THIS section's own
            # Research Notes, drop any that duplicate a pointer already used
            # by an earlier section (literal + semantic check —
            # filter_novel_pointers), and tell the model to cover exactly
            # those. That's a positive, concrete task ("write about these
            # specific facts") instead of an abstract one to avoid, and it
            # never hands the model another section's prose at all — nothing
            # left to copy. Needs real source material to draw pointers
            # from; a sub-topic with no Research Notes falls through to the
            # plain prompt below unchanged.
            if has_context:
                pointer_prompt = f"""<start_of_turn>user
Research Notes:
{state.get('current_context', '')}{ref_block}

List 4-6 distinct fact pointers (5-8 words each) about "{item['sub_topic']}" from the Research Notes above only. Include the specific named entities (people, organizations, places) and dates the Research Notes actually give — a pointer naming who/what/when is more useful than a vague one. {_date_grounding_clause(topic)}
One line per pointer, exactly this format, nothing else:
<pointer text> | [<citation number, or none>]
Example (different topic, format only — never reuse this example's topic, wording, or number in your answer): EU battery recycling rate rising sharply | [2]
<end_of_turn>
<start_of_turn>model
"""
                pointer_response = llm.invoke(pointer_prompt, max_tokens=200)
                parsed = _parse_pointer_lines(pointer_response)

                # Grounding checks — code-level, not prompt-level, for the
                # same reason duplication ended up being enforced in code:
                # a pointer that fails these is dropped outright rather than
                # trusted to self-correct on instruction alone.
                context_text = state.get("current_context", "")
                candidates = []
                for c in parsed:
                    if not _valid_pointer_year(c["text"], topic, context_text):
                        print(f"write_node: dropping pointer with a wrong/unsupported year for "
                              f"{item['sub_topic']!r}: {c['text']!r}")
                        continue
                    bad_entities = _ungrounded_entities(c["text"], context_text)
                    if bad_entities:
                        print(f"write_node: dropping pointer with ungrounded entity {bad_entities} for "
                              f"{item['sub_topic']!r}: {c['text']!r}")
                        continue
                    candidates.append(c)

                # A pointer's citation number is only trustworthy if it's
                # actually one of the numbers assigned to a real source in
                # THIS item's Research Notes (research_node labels each
                # source's lines "[N] Title: ...") — otherwise strip the
                # number rather than let a misattributed citation through;
                # the fact itself may still be genuine even if the model
                # attached the wrong source number to it.
                valid_citation_numbers = {int(n) for n in re.findall(r"\[(\d+)\]", context_text)}
                for c in candidates:
                    if c["citation"] is not None and c["citation"] not in valid_citation_numbers:
                        print(f"write_node: stripping invalid citation [{c['citation']}] from pointer "
                              f"{c['text']!r}")
                        c["citation"] = None

                existing_texts = [p["text"] for p in state.get("used_pointers", [])]
                novel_texts = set(filter_novel_pointers([c["text"] for c in candidates], existing_texts))
                novel_pointers = [c for c in candidates if c["text"] in novel_texts]
                dropped_texts = [c["text"] for c in candidates if c["text"] not in novel_texts]
                print(f"write_node: pointers for {item['sub_topic']!r} — "
                      f"kept: {[p['text'] for p in novel_pointers]}"
                      + (f" | dropped as duplicate: {dropped_texts}" if dropped_texts else ""))

            pointer_clause = ""
            if novel_pointers:
                pointer_lines = "\n".join(
                    f"- {p['text']}" + (f" [{p['citation']}]" if p.get("citation") else "")
                    for p in novel_pointers
                )
                pointer_clause = f"\n\nCover each of these facts, citing each with its bracketed number:\n{pointer_lines}\n"

            prompt = f"""<start_of_turn>user
Write one detailed paragraph about "{item['sub_topic']}" (topic: "{topic}") using only facts from the Research Notes below. No bullet points, no "Firstly/Secondly" structure, no generic opening like "X plays a key role" — start with a specific fact. Output ONLY the paragraph itself — no lead-in like "Sure, here is..." or "Here's the paragraph:".{date_clause}{citation_clause}{pointer_clause}{revision_clause}

Research Notes:
{state.get('current_context', '')}{ref_block}
<end_of_turn>
<start_of_turn>model
"""
        # LLM output sometimes contains doubled spaces (e.g. "faster ,and" or
        # a stray "  " between words) — collapse runs of horizontal
        # whitespace without touching the paragraph's own line breaks.
        content = llm.invoke(prompt, max_tokens=max_tokens).strip()
        content = strip_llm_preamble(content)
        content = fix_missing_spaces(content)
        content = re.sub(r"[ \t]{2,}", " ", content)
        content = re.sub(r"[ \t]+([.,;:!?])", r"\1", content)
        content = re.sub(r"([.,;:!?]){2,}", r"\1", content)
        draft: SectionRecord = {
            "kind": item["kind"], "theme": item["theme"], "sub_topic": item.get("sub_topic"),
            "content": content, "weightage": item.get("weightage"),
            "sources": state.get("current_sources", []) if item["kind"] == "standard" else [],
            "media": [], "status": "pending", "validation_notes": None,
            "revision_count": item.get("revision_count", 0),
        }
        return {"current_section_draft": draft, "current_pointers": novel_pointers}

    # Output from this node goes to the validate_node.
    return write_node


def make_validate_node(llm: LlamaCpp, vision_checker):
    """Strips self-duplicate sentences (repeated within the same section) and
    cross-section duplicate sentences (extending or dropping the section if
    too little survives), then checks text (copyright overlap against
    source material, factual grounding) and, for SYSTEM-GENERATED media only
    (never user-supplied), a copyright/SFW check via the vision model."""

    def validate_node(state: ResearchState) -> dict:
        draft = state["current_section_draft"]
        item = state["work_queue"][0]
        issues = []

        # Year-grounding is enforced in code here, not just in the write
        # prompt — _valid_pointer_year (make_write_node) only guards the
        # pointer list, but the model still reads the FULL raw Research
        # Notes when writing the actual paragraph and can pull a wrong year
        # from there directly, and a meta section has no pointers to guard
        # at all. Observed directly: a 2025-topic report's Overview (meta,
        # no pointers) and a standard section's final prose both narrated
        # events as happening "in 2015." Corrected deterministically rather
        # than asking for another rewrite — see _correct_wrong_years.
        corrected = _correct_wrong_years(draft["content"], state["topic"])
        if corrected != draft["content"]:
            print(f"validate_node: corrected wrong year(s) in {item.get('sub_topic') or item['theme']!r}")
        draft = {**draft, "content": corrected}

        # Duplication across sections is now caught and fixed in code rather
        # than by asking the model not to repeat itself — that instruction
        # (in any phrasing tried: a bulleted "don't restate this" list, or
        # framing it as "continue from this text") wasn't reliably followed,
        # and a blind full-prompt retry often just reproduced the same
        # duplicate again. Instead: strip out any sentence that's actually a
        # near-duplicate of something already written; if too little
        # survives, ask ONE narrow follow-up for a couple more sentences and
        # dedupe again; if that still isn't enough, there's genuinely
        # nothing new left to say, so the section is dropped rather than
        # published thin or repetitive. Applies to every item kind — meta
        # sections duplicate each other too (see GRAPH.md).
        # A section can also repeat ITSELF within one paragraph (observed
        # directly: "poor positioning of engines and single sensor
        # malfunction" stated near-verbatim twice in one section) — cross-
        # section dedup below can't catch that since it only ever compares
        # against OTHER sections. Strip self-duplicates first.
        prior_sections_text = "\n".join(s["content"] for s in state.get("sections", []) if s.get("content"))
        self_deduped = drop_self_duplicate_sentences(draft["content"])
        if self_deduped != draft["content"].strip():
            print(f"validate_node: removed self-duplicate sentence(s) within "
                  f"{item.get('sub_topic') or item['theme']!r} — kept {len(split_sentences(self_deduped))}/"
                  f"{len(split_sentences(draft['content']))} sentence(s)")
        kept = drop_duplicate_sentences(self_deduped, prior_sections_text)
        if kept != self_deduped:
            print(f"validate_node: removed duplicate sentence(s) from {item.get('sub_topic') or item['theme']!r} "
                  f"— kept {len(split_sentences(kept))}/{len(split_sentences(self_deduped))} sentence(s)")
        if len(split_sentences(kept)) < 3:
            addition = _extend_content(llm, state, item, kept)
            addition = _correct_wrong_years(addition, state["topic"])
            combined = f"{kept} {addition}".strip() if kept else addition
            combined = drop_self_duplicate_sentences(combined)
            kept = drop_duplicate_sentences(combined, prior_sections_text)
            print(f"validate_node: requested additional content for "
                  f"{item.get('sub_topic') or item['theme']!r} — {len(split_sentences(kept))} sentence(s) "
                  "survive after retry + dedup")
            if len(split_sentences(kept)) < 3:
                print(f"validate_node: eliminating {item.get('sub_topic') or item['theme']!r} — still fewer "
                      "than 3 non-duplicate sentences after one retry; nothing new left to say.")
                return {
                    "work_queue": state["work_queue"][1:],
                    "last_theme_written": item["theme"],
                    "current_section_draft": None, "current_media": [], "current_pointers": [],
                }
        draft = {**draft, "content": kept}

        overlap = _max_overlap_ratio(draft["content"], state.get("current_context", ""))
        if overlap > COPYRIGHT_OVERLAP_THRESHOLD:
            issues.append(("copyright", f"{overlap:.0%} overlap with source text — rephrase in your own words."))

        # Deterministic prose-quality checks — code-level, same reasoning as
        # _correct_wrong_years/drop_duplicate_sentences above: these failure
        # modes (see open-issues.md) weren't reliably prevented by prompt
        # instructions alone, so they're detected here and sent back for a
        # targeted rewrite instead.
        spliced_titles = _spliced_source_titles(draft["content"], draft.get("sources", []))
        if spliced_titles:
            issues.append(("title_splice",
                            f"Paraphrase this fact instead of copying the source title verbatim: {spliced_titles}"))

        if _PLACEHOLDER_RE.search(draft["content"]):
            issues.append(("placeholder",
                            "Remove the empty '( )%'-style placeholder — cite a real figure or drop the claim."))

        quarter_hits = _invalid_quarter_claims(draft["content"])
        if quarter_hits:
            issues.append(("fabrication", f"Invalid quarter reference (quarters only run 1-4): {quarter_hits}"))

        if _MALFORMED_YEAR_CITATION_RE.search(draft["content"]):
            issues.append(("fabrication",
                            "A citation bracket '[N]' appears where a year was expected — "
                            "write an actual year or drop the claim."))

        if draft["kind"] == "standard":
            combined_context = (state.get("current_context", "") or "") + "\n" \
                + (state.get("current_reference_context", "") or "")
            valid_citation_numbers = {int(n) for n in re.findall(r"\[(\d+)\]", combined_context)}
            invalid_cites = _find_invalid_citation_numbers(draft["content"], valid_citation_numbers)
            if invalid_cites:
                issues.append(("citation",
                                f"Citation number(s) {invalid_cites} don't match any source in the Research "
                                "Notes — cite a real number or drop the bracket."))
            elif valid_citation_numbers:
                miscite_hits = _find_miscited_sentences(
                    draft["content"], _context_blobs(state.get("current_context", ""))
                )
                if miscite_hits:
                    issues.append(("citation", "Possible miscitation — " + "; ".join(miscite_hits)))

        if draft["kind"] == "standard" and state.get("current_context") and not issues:
            check_prompt = f"""<start_of_turn>user
Research Notes:
{state['current_context']}

Section to check:
{draft['content']}

Does the section contain any claim that is NOT supported by the Research Notes?
Answer with exactly "OK" if fully supported, or a one-sentence description of the unsupported claim.
<end_of_turn>
<start_of_turn>model
"""
            verdict = llm.invoke(check_prompt, max_tokens=80).strip()
            if not verdict.upper().startswith("OK"):
                issues.append(("factual", verdict))

        media = state.get("current_media", []) or []
        # Vision-model check bypassed on request — system-generated media is
        # accepted without an SFW/copyright pass. vision_checker is still
        # accepted as a parameter (and passed a fake in tests) so this is a
        # one-line revert if the check needs to come back.

        # route_after_validate already routes a "media" revision_kind back to
        # non_text_master_node to retry a failed generated item (see its
        # docstring and GRAPH.md) — but nothing here ever actually produced
        # that issue, so a diagram/table/photo that failed on its one attempt
        # (make_diagram_content's Mermaid check rejecting it, or
        # internet_image_content/make_data_table_content coming back with
        # neither a path nor markup) went straight into the approved section
        # as a permanent placeholder-comment. Same is_placeholder condition
        # format_sections_node (nodes_output.py) uses to decide "this became
        # a TODO comment" — a text issue (copyright/factual) still takes
        # priority when both are present, same as before, since only one
        # revision_kind can be acted on per loop and this is appended last.
        # User-supplied media is never retried — retrying would just re-read
        # the same file and get the same result.
        if not issues:
            failed_generated = [
                m for m in media
                if m.get("source") == "generated"
                and (m.get("status") == "rejected" or not (m.get("path") or m.get("markup")))
            ]
            if failed_generated:
                reasons = "; ".join(
                    m.get("validation_notes") or f"{m.get('type')} generation failed"
                    for m in failed_generated
                )
                issues.append(("media", reasons))

        revision_count = item.get("revision_count", 0)
        if not issues or revision_count >= MAX_REVISIONS:
            final: SectionRecord = {
                **draft, "media": media,
                "status": "approved" if not issues else "approved_with_warning",
                "validation_notes": "; ".join(f"{k}: {v}" for k, v in issues) or None,
            }
            return {
                "sections": state["sections"] + [final],
                "used_pointers": state.get("used_pointers", []) + state.get("current_pointers", []),
                "work_queue": state["work_queue"][1:],
                "last_theme_written": item["theme"],
                "current_section_draft": None, "current_media": [], "current_pointers": [],
            }

        kind, note = issues[0]
        next_item: WorkItem = {**item, "revision_note": note, "revision_kind": kind,
                                "revision_count": revision_count + 1}
        return {"work_queue": [next_item] + state["work_queue"][1:],
                "current_section_draft": None, "current_media": []}
    # Output from this node goes to route_after_validate() to decide route.
    return validate_node

# ==========================================
# ROUTES
# ==========================================

def route_next(state: ResearchState) -> Literal["research", "write", "finalize"]:
    """If the work queue is empty, route to assign_citations_node; otherwise route to make_research_node if the next item is standard, or make_write_node if it's meta."""
    if not state["work_queue"]:
        return "finalize"
    return "research" if state["work_queue"][0]["kind"] == "standard" else "write"

def route_after_research(state: ResearchState) -> Literal["non_text_master", "write"]:
    """Routes to non_text_master_node whenever the item has ANY content
    requests at all — not just ones needing a real fetch. non_text_master_node
    itself already handles both cases (calling a generator, or reading a
    user-supplied file straight off disk); a request with user_supplied_path
    still has to pass through it to actually land in current_media, otherwise
    a fully user-supplied section silently ends up with no media at all."""
    item = state["work_queue"][0]
    return "non_text_master" if item.get("content_requests") else "write"

def route_after_validate(state: ResearchState) -> Literal["research", "write", "non_text_master", "finalize"]:
    """If the current work item was approved, route to the next item in the work queue (research/write/finalize).
    If it was rejected, route to research/write/non_text_master depending on the revision_kind
    ("copyright", "title_splice", "placeholder", "fabrication", "citation", and meta items all go to write —
    phrasing/citation fixes that don't need new source material; "factual" goes to research to get better
    source material; "media" goes to non_text_master). Cross-section duplication is handled separately, in
    code, inside validate_node itself — it never produces a revision_note/revision_kind."""

    item = state["work_queue"][0] if state["work_queue"] else None
    if item and item.get("revision_note"):
        kind = item.get("revision_kind")
        if kind == "media":
            return "non_text_master"
        if kind in ("copyright", "title_splice", "placeholder", "fabrication", "citation") or item["kind"] == "meta":
            return "write"
        return "research"
    return route_next(state)
