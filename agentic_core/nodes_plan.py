"""Nodes — make_plan_node, review_outline_node, build_work_queue_node.

Human-in-the-loop: review_outline_node uses LangGraph's interrupt() + a
checkpointer. The RESUME payload must be JSON-serializable, so a
user-uploaded zip of content files is NOT passed through Command(resume=...)
— the caller (Streamlit) saves and extracts the zip to
`<outline_dir>/contents/` BEFORE resuming, and only a manifest travels
through resume:
    Command(resume={
        "outline": [...],
        "user_provided_content": {"cheese_map.png": {"theme": "...", "sub_topic": "...", "type": "internet-image"}},
    })
"""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Dict, List

from langchain_community.llms import LlamaCpp
from langgraph.types import interrupt

from .config import ReportConfig
from .constants import DATA_DIR, MAX_CONTENT_REQUESTS_PER_THEME
from .state import ResearchState, WorkItem

# ==========================================
# NODES
# ==========================================

def _parse_sub_topic_lines(text: str, theme: str = "") -> List[dict]:
    """Parses 'Sub-topic Name | query one | query two' lines from the LLM.

    Nested JSON was tried first and dropped: a quantized 7B instruct model
    reliably mangles it — smart/curly quotes, //  and # comments dropped mid-
    object, wrong key names ("sub-topic" vs "sub_topic"), missing commas,
    unbalanced brackets — a different failure shape almost every run, with no
    end to the list of special cases worth chasing. A flat pipe-delimited
    line is something small models reproduce far more reliably, and needs no
    real parser: split on "|", strip stray bullets/quotes/markdown per field.

    content_requests is intentionally not requested via this format — it
    doesn't fit a single line without reintroducing nested syntax the model
    would mangle the same way. It's always empty here.

    theme, when given, filters out a sub-topic that's just the theme name
    echoed back verbatim AND carries no real queries (e.g. asked about
    "KEY INSIGHTS", answered with a line reading "KEY INSIGHTS |" and nothing
    after it) — observed directly: that combination is a technically-
    parseable item with zero real content, worse than the per-theme fallback
    plan_node uses when nothing parses at all. A theme-name title WITH real
    queries attached (also observed directly: "**Recommendations | Market
    segmentation based on car type demand, ...") is kept — the model reusing
    the theme name as a lazy label doesn't mean the content behind it is
    fake, and discarding it would throw away real, useful queries.

    Also handles a second observed shape: the title on its own line (often
    numbered/bolded — "1. **Market Share Trends by Car Type:** description
    text...") followed by the queries on the NEXT line starting with a bare
    leading "|" and no title of its own. Tracks the most recent title-shaped
    line as a "pending" title and attaches it to the next leading-pipe line
    that has no title in its own first field — this is what let a
    completely genuine sub-topic+queries pair survive instead of being
    dropped twice over (once for the untitled query line, once for the
    unpiped title line).
    """
    theme_key = _normalize_theme(theme) if theme else None
    items = []
    pending_title = None
    for raw_line in (text or "").splitlines():
        line = raw_line.strip().strip("`")
        if not line:
            continue
        if "|" in line:
            parts = [_clean_field(p) for p in line.split("|")]
            sub_topic = _clean_sub_topic(parts[0])
            queries = [p for p in parts[1:] if p][:3]
            if not sub_topic and pending_title:
                sub_topic = pending_title
            pending_title = None
            is_bare_theme_echo = (
                theme_key is not None and _normalize_theme(sub_topic) == theme_key and not queries
            )
            if (sub_topic and re.search(r"[A-Za-z]", sub_topic)
                    and not _is_placeholder_sub_topic(sub_topic)
                    and not is_bare_theme_echo):
                items.append({"sub_topic": sub_topic, "queries": queries, "content_requests": []})
        elif _TITLE_CANDIDATE_RE.match(line):
            candidate = _clean_sub_topic(line)
            if (candidate and re.search(r"[A-Za-z]", candidate)
                    and not _is_placeholder_sub_topic(candidate)
                    and not (theme_key is not None and _normalize_theme(candidate) == theme_key)):
                pending_title = candidate
    return items


_MAX_SUB_TOPIC_CHARS = 70
_PLACEHOLDER_SUB_TOPIC_RE = re.compile(r"^sub\s*-?\s*topic\s*(name|\d+)?$", re.IGNORECASE)
_INVISIBLE_CHARS_RE = re.compile(r"[​‌‍﻿]")
# A line with no "|" is only worth tracking as a possible title for the NEXT
# leading-pipe queries-only line when it actually looks like one — numbered
# ("1. Title") and/or bolded ("**Title") — not just any prose continuation
# line, which would otherwise get wrongly paired with an unrelated later line.
_TITLE_CANDIDATE_RE = re.compile(r"^[\s\-]*(\d+[.)]\s*)?\*{1,2}\S")


def _clean_field(raw: str) -> str:
    """Applied to every pipe-delimited field (title and each query): strips
    stray HTML tags (<strong>, <b>, <br>, ...), invisible Unicode artifacts
    (BOM, zero-width space) and surrounding quote/asterisk/space noise. Small
    models emit tags in either field, not just the title."""
    s = re.sub(r"<[^>]*>", " ", raw)
    s = _INVISIBLE_CHARS_RE.sub("", s)
    return s.strip(" \"'*")


def _clean_sub_topic(raw: str) -> str:
    """Handles Gemma ignoring the "just a short name" instruction and writing
    a full descriptive blurb into the field instead, using its own ad-hoc
    ":"/"**"/bullet formatting rather than the requested "|" — kept short by
    truncating at the first sentence/colon break, hard-capped at
    _MAX_SUB_TOPIC_CHARS if that break doesn't actually shorten it."""
    s = raw.replace("**", "")
    s = re.sub(r"^[\s\-*\d.)]+", "", s)
    s = re.sub(r"[\s\-*]+$", "", s)
    if len(s) > _MAX_SUB_TOPIC_CHARS:
        head = re.split(r"[:.]", s, 1)[0].strip()
        if not head or len(head) > _MAX_SUB_TOPIC_CHARS:
            head = s[:_MAX_SUB_TOPIC_CHARS].rsplit(" ", 1)[0].strip()
        s = head
    # A trailing comma/colon/semicolon means the model got cut off mid-phrase
    # ("...worldwide market size ,") — trailing punctuation on a title is
    # always wrong regardless of length, so it's dropped unconditionally.
    s = re.sub(r"[\s,;:\-]+$", "", s)
    return s


def _is_placeholder_sub_topic(sub_topic: str) -> bool:
    """Small models sometimes echo the prompt's own example line back
    verbatim instead of substituting real content — catch "Sub-topic Name",
    "Sub-topic 1", "Sub - topic  2", etc. so that line is treated the same as
    an unparseable one rather than becoming a real (garbage) section."""
    return bool(_PLACEHOLDER_SUB_TOPIC_RE.match(sub_topic))


def make_plan_node(llm: LlamaCpp, report_cfg: ReportConfig):
    """Generates draft_outline (theme/sub_topic/queries) and persists it to
    <outline_dir>/outline_draft.json."""

    def plan_node(state: ResearchState) -> dict:
        """Requests sub-topics ONE THEME PER LLM CALL, rather than one call
        covering every theme. A quantized 7B instruct model asked for a large
        multi-theme response in one shot reliably fills in only the first
        theme and drops the rest — observed directly against real Gemma
        output, not a hypothetical. Splitting per-theme also removes the need
        to match the model's returned "theme" string back to a canonical
        theme (a second source of failure with small models): the theme is
        assigned by us, not parsed from the response."""
        topic, report_format = state["topic"], state["report_format"]
        themes = report_cfg.themes_for_format(report_format)
        standard_themes = report_cfg.get_standard_themes(report_format)

        ref_docs = state.get("reference_docs", {})
        ref_digest = ""
        ok_refs = {f: m for f, m in ref_docs.items() if m.get("status") == "ok"}
        if ok_refs:
            ref_digest = "\nReference material provided (use to inform sub-topics where relevant): " + \
                ", ".join(f"{f} ({m['char_count']} chars)" for f, m in ok_refs.items())

        sub_topics, seen = [], set()
        for theme in standard_themes:
            theme_prompt = f"""<start_of_turn>user
Topic: {topic}
Report format: {report_format}
Theme: {theme}{ref_digest}
List 1 to 3 sub-topics for a professional report on '{topic}', specifically under the theme "{theme}". Order them in a logical hierarchy.
Output EXACTLY one line per sub-topic, in this format and nothing else — no numbering, no bullets, no quotes, no JSON, no HTML tags, no colon, no explanation. Each sub-topic name must be a short noun phrase of at most 8 words:
<short sub-topic title> | <query 1> | <query 2> | <query 3>
Example line, about a completely different topic, showing the format only — do not reuse this example's wording:
Battery Recycling Regulations | EU battery recycling law | lithium battery recycling policy
Use up to 3 short web-search query suffixes per sub-topic. Do NOT include "{topic}" in the queries.
<end_of_turn>
<start_of_turn>model
"""
            response = llm.invoke(theme_prompt)
            raw_items = _parse_sub_topic_lines(response, theme=theme)
            if not raw_items:
                print(f"plan_node: could not parse sub-topics for theme {theme!r} — "
                      "using a single generic sub-topic. Raw response was:\n"
                      f"{response!r}\n")
                raw_items = [{"sub_topic": theme.title(), "queries": [f"{topic} {theme.lower()}"],
                              "content_requests": []}]

            for item in raw_items:
                name = item["sub_topic"].strip().lower()
                if name and name not in seen:
                    seen.add(name)
                    item["theme"] = theme
                    item["queries"] = list({f"{topic} {q}".strip() for q in item.get("queries", [])})
                    sub_topics.append(item)

        run_id = state.get("run_id") or uuid.uuid4().hex[:8]
        outline_dir = os.path.join(DATA_DIR, run_id)
        os.makedirs(outline_dir, exist_ok=True)
        with open(os.path.join(outline_dir, "outline_draft.json"), "w", encoding="utf-8") as f:
            json.dump(sub_topics, f, indent=2)

        return {"themes": themes, "standard_themes": standard_themes,
                "draft_outline": sub_topics, "run_id": run_id, "outline_dir": outline_dir}

    # Output from this node goes to the review_outline_node.
    return plan_node


def review_outline_node(state: ResearchState) -> dict:
    """Pauses via interrupt() and hands the draft outline to the caller. Themes
    are shown but NOT editable — only sub_topics/queries/content_requests under
    them, plus an optional weightage.

    Resume with a JSON-serializable dict:
        Command(resume={"outline": [...], "user_provided_content": {filename: {...}}})
    The zip itself must already be saved+extracted to <outline_dir>/contents/
    by the caller before resuming — raw file bytes never travel through resume.
    """
    payload = {
        "themes": state["themes"], "standard_themes": state["standard_themes"],
        "draft_outline": state["draft_outline"],
        "instructions": (
            "Edit sub_topics/queries/content_requests under the existing themes "
            "only (do not add new themes). Optionally set 'weightage' (0-100) "
            "per item; unset items split the remaining weight evenly. To supply "
            "your own content for a request, upload a zip and reference the "
            "filename via 'user_provided_content'."
        ),
    }
    edited = interrupt(payload)
    outline = edited.get("outline") if isinstance(edited, dict) else edited
    outline = outline if isinstance(outline, list) and outline else state["draft_outline"]
    user_content = edited.get("user_provided_content", {}) if isinstance(edited, dict) else {}

    weights = [it.get("weightage") for it in outline]
    given_total = sum(w for w in weights if isinstance(w, (int, float)))
    unset_count = sum(1 for w in weights if not isinstance(w, (int, float)))
    if unset_count:
        remaining = max(0.0, 100.0 - given_total)
        share = remaining / unset_count if remaining > 0 else 100.0 / max(len(outline), 1)
        for it in outline:
            if not isinstance(it.get("weightage"), (int, float)):
                it["weightage"] = round(share, 2)

    with open(os.path.join(state["outline_dir"], "outline.json"), "w", encoding="utf-8") as f:
        json.dump(outline, f, indent=2)

    # Output from this node goes to the build_work_queue_node.
    return {"reviewed_outline": outline, "user_provided_content": user_content}


def _normalize_theme(s: str) -> str:
    """Collapses case/punctuation/whitespace differences (e.g. 'Detailed
    Analysis:' vs 'DETAILED ANALYSIS') so matching an LLM-returned theme
    string against the canonical theme list doesn't silently fail on
    formatting noise — see build_work_queue_node."""
    return re.sub(r"[^A-Z0-9]+", " ", (s or "").upper()).strip()


def build_work_queue_node(state: ResearchState) -> dict:
    """Flattens themes + the reviewed outline into an ordered work queue,
    resolving content_requests against any user-supplied files."""
    topic = state["topic"]
    themes, standard_themes = state["themes"], state["standard_themes"]
    reviewed = state["reviewed_outline"]
    user_content = state.get("user_provided_content", {})

    work_queue: List[WorkItem] = []
    for theme in themes:
        if theme not in standard_themes:
            work_queue.append({"kind": "meta", "theme": theme, "content_requests": [], "revision_count": 0})
            continue
        theme_key = _normalize_theme(theme)
        theme_items = [s for s in reviewed if _normalize_theme(s.get("theme", "")) == theme_key]
        if not theme_items:
            theme_items = [{"sub_topic": theme, "queries": [f"{topic} {theme.lower()}"],
                             "weightage": 100.0 / max(len(standard_themes), 1), "content_requests": []}]
        for s in theme_items:
            requests_copy = [dict(r) for r in (s.get("content_requests") or [])]
            for fname, meta in user_content.items():
                if meta.get("theme") == theme and meta.get("sub_topic") == s.get("sub_topic"):
                    for req in requests_copy:
                        if req.get("type") == meta.get("type") and not req.get("user_supplied_path"):
                            req["user_supplied_path"] = os.path.join(state["outline_dir"], "contents", fname)
            work_queue.append({
                "kind": "standard", "theme": theme,
                "sub_topic": s.get("sub_topic", "General Section"),
                "queries": s.get("queries", []),
                "weightage": s.get("weightage", 0.0),
                "content_requests": requests_copy,
                "revision_count": 0,
            })

    # Output from this node goes to make_content_requests_node. citation_map/
    # citations start empty here and are built up incrementally by
    # research_node as sources are first discovered — see its docstring.
    return {"work_queue": work_queue, "sections": [], "seen_urls": [],
            "last_theme_written": None, "current_section_draft": None, "current_media": [],
            "current_pointers": [], "used_pointers": [],
            "citation_map": {}, "citations": []}


_VALID_CONTENT_TYPES = {"internet-image", "data-table", "diagram"}


def add_content_request(outline: List[dict], sub_topic: str, content_type: str,
                         description: str, user_supplied_path: str | None = None,
                         diagram_type: str | None = None) -> dict:
    """Attaches a content request to the outline item matching sub_topic —
    the standard way to add non-text content by hand (during outline review,
    in a test script, from a future Streamlit form), instead of hand-indexing
    into draft_outline and constructing the {"type", "description", ...}
    dict yourself every time. Call it on the draft_outline you get back from
    the review_outline_node interrupt, before resuming.

    diagram_type only applies when content_type == "diagram" — one of
    content.py's DIAGRAM_CONFIGS keys (flowchart/sequenceDiagram/
    classDiagram/mindmap); defaults to "flowchart" if left unset, matching
    what content_requests_node's automatic (LLM-driven) decisions always
    produce, since that prompt never picks a sub-type on its own.

    Raises ValueError (with the actual available sub_topics, or the valid
    type list) rather than silently doing nothing on a typo — a manually
    added content request that's silently ignored is a confusing thing to
    debug later.

    Returns the outline item that was modified, for convenience (e.g. to
    inspect or further edit it).
    """
    if content_type not in _VALID_CONTENT_TYPES:
        raise ValueError(f"content_type must be one of {sorted(_VALID_CONTENT_TYPES)}, got {content_type!r}")
    if diagram_type is not None:
        if content_type != "diagram":
            raise ValueError(f"diagram_type only applies when content_type is 'diagram', got {content_type!r}")
        from .content import DIAGRAM_CONFIGS
        if diagram_type not in DIAGRAM_CONFIGS:
            raise ValueError(f"diagram_type must be one of {sorted(DIAGRAM_CONFIGS)}, got {diagram_type!r}")

    matches = [item for item in outline
               if (item.get("sub_topic") or "").strip().lower() == sub_topic.strip().lower()]
    if not matches:
        available = [item.get("sub_topic") for item in outline]
        raise ValueError(f"No outline item with sub_topic={sub_topic!r}. Available: {available}")
    if len(matches) > 1:
        raise ValueError(f"Multiple outline items match sub_topic={sub_topic!r} — "
                          "rename one of them or edit content_requests directly.")

    request = {"type": content_type, "description": description}
    if user_supplied_path:
        request["user_supplied_path"] = user_supplied_path
    if diagram_type:
        request["diagram_type"] = diagram_type

    item = matches[0]
    item.setdefault("content_requests", []).append(request)
    return item


_CONTENT_TYPE_RE = re.compile(
    r"internet[- ]image|data[- ]table|diagram|photo(?:graph)?|\bimage\b", re.IGNORECASE
)
# content_requests_node's prompt below calls the image option "a
# representative photo" three times and only gives the literal type keyword
# "internet-image" once, at the very end — a small model asked to reply with
# just the type word is at least as likely to echo "photo" (the term it was
# given prominently) as the technical "internet-image" one. Without this
# alias, _CONTENT_TYPE_RE never matched "photo" at all, so every such reply
# fell through to "no valid type keyword found" and was silently treated as
# a decline — observed indirectly as image content requests never appearing
# in any generated report, across multiple different topics.
_CONTENT_TYPE_ALIASES = {"photo": "internet-image", "photograph": "internet-image", "image": "internet-image"}
_MAX_CONTENT_DESCRIPTION_CHARS = 100
# A remainder that starts like this is the model still reasoning about the
# PROMPT'S OWN CONDITIONS ("If this sub-topic involves any numbers...", "or
# a representative photo...") rather than giving a concrete description of
# what to actually generate — observed directly as garbage descriptions like
# "** If statistics are involved" and "or representative photo based on its
# content and purpose" leaking into the final report's TODO placeholders.
_META_DESCRIPTION_RE = re.compile(
    r"^(if\b|or\b|and\b|would\b|say so\b|based on\b|otherwise\b)", re.IGNORECASE
)


def _parse_content_request_line(text: str) -> dict | None:
    """Parses the 'none' / 'type | description' response — same robust
    line-format approach as _parse_sub_topic_lines, deliberately NOT a JSON
    call, for the same reliability reasons documented there.

    The model routinely ignores the requested format, in more than one way
    observed against real output:
      - bare type on its own line, justification as prose afterward
        (e.g. "Data-table.\n\nThe subtopic involves...")
      - an extra leading field before the real answer — e.g. echoing the
        sub-topic name first: "Marketing Strategies | Data-table. ..."
        (this one was being silently declined: the old parser only checked
        the segment BEFORE the first "|", which is exactly where the model
        put the noise, not the type)
    Rather than pattern-matching each new shape, this searches the WHOLE
    response for a valid type keyword wherever it actually appears, and
    treats everything before it as noise (sub-topic echo, numbering,
    whatever). "none" is still checked first and only against the first
    line, so a legitimate decline that merely *mentions* one of these words
    while explaining itself ("does not need a diagram") isn't misread as an
    approval.
    """
    raw = (text or "").strip().strip("`")
    if not raw:
        return None
    first_line = raw.splitlines()[0].strip() if raw.splitlines() else ""
    if first_line.lower().startswith("none"):
        return None

    match = _CONTENT_TYPE_RE.search(raw)
    if not match:
        return None
    matched = re.sub(r"[ _]", "-", match.group(0).lower())
    ctype = _CONTENT_TYPE_ALIASES.get(matched, matched)

    # Whatever follows the matched keyword — trim leading punctuation/pipes
    # left over from the model's own formatting attempt, then the same
    # markdown/HTML/quote-noise cleanup _parse_sub_topic_lines uses (strips
    # a leading "**" left over from the model echoing the prompt's own
    # bolded conditions, among other formatting noise).
    remainder = raw[match.end():].lstrip(" .:|\n").strip()
    remainder_first_line = _clean_field(remainder.splitlines()[0]) if remainder else ""
    # A short remainder is a real description (the "| description" case). A
    # long one is the model's justification prose ("...would be the most
    # helpful format for this section of text to provide information on...")
    # — not search-worthy, so it's discarded. Likewise a remainder that reads
    # as a fragment of the PROMPT'S OWN conditional wording rather than a
    # concrete description (_META_DESCRIPTION_RE) is discarded rather than
    # kept — either way make_content_requests_node fills in a clean
    # description from the sub-topic it already knows.
    description = (
        remainder_first_line
        if 0 < len(remainder_first_line) <= _MAX_CONTENT_DESCRIPTION_CHARS
        and not _META_DESCRIPTION_RE.match(remainder_first_line)
        else None
    )
    return {"type": ctype, "description": description}


def make_content_requests_node(llm: LlamaCpp):
    """One-time pass, right after plan_node builds draft_outline and BEFORE
    review_outline_node's interrupt, deciding which sub-topics get non-text
    content. Runs here — not after review — so the auto-decided requests are
    already part of the outline the human reviewer sees: the caller can
    render a media-upload row for each one and let the user attach a file
    before resuming, the same way a manually-added content_request already
    could. Asks per sub-topic in a single short line — not one big JSON call
    listing every theme/sub-topic, which would need the LLM to correctly echo
    back theme/sub_topic identifiers to say where each request belongs (the
    exact kind of matching fragility plan_node's outline step was rewritten
    to avoid).

    Capped at MAX_CONTENT_REQUESTS_PER_THEME sub-topics per theme, taken in
    the order plan_node produced them (its per-theme prompt already asks for
    a logical hierarchy, and weightage isn't assigned until review_outline_
    node runs afterward, so there's nothing else to rank by yet) — not a
    target to hit, the model can decline every one of them. Sub-topics that
    already carry a content_request (e.g. added by hand into draft_outline
    before this node runs) are left untouched.

    A sub-topic the user adds DURING outline review (after this node has
    already run) gets no automatic content_request — the user must add one
    explicitly (see add_content_request) if they want non-text content for
    it, since there's no later pass that revisits the outline."""

    def content_requests_node(state: ResearchState) -> dict:
        draft_outline = state["draft_outline"]
        topic = state["topic"]

        by_theme: Dict[str, List[int]] = {}
        for idx, item in enumerate(draft_outline):
            by_theme.setdefault(item.get("theme", ""), []).append(idx)

        for theme, indices in by_theme.items():
            for idx in indices[:MAX_CONTENT_REQUESTS_PER_THEME]:
                item = draft_outline[idx]
                if item.get("content_requests"):
                    continue
                prompt = f"""<start_of_turn>user
Topic: {topic}
Theme: {theme}
Sub-topic: {item.get('sub_topic')}
Decide whether this sub-topic would benefit from ONE of: a data-table (it involves numbers/statistics/figures), a diagram (it involves a sequence of steps or a process), or a representative photo (something concrete a reader could picture) — or none of these.
Reply with EXACTLY ONE line, nothing else — no explanation, no restating these rules:
type | short, concrete description of the specific content to generate
The description must name the actual thing (e.g. what the table/diagram/photo shows) — never restate or paraphrase this instruction's own wording.
Example (different topic, format only): data-table | Battery recycling rates by country, 2020-2025
Only reply "none" if truly nothing above would add value.
Valid types: internet-image, data-table, diagram
<end_of_turn>
<start_of_turn>model
"""
                response = llm.invoke(prompt, max_tokens=60)
                req = _parse_content_request_line(response)
                if req:
                    if not req["description"]:
                        req["description"] = f"{item.get('sub_topic')} in {topic}"
                    item["content_requests"] = [req]
                    print(f"content_requests_node: approved {req['type']} for "
                          f"{item.get('sub_topic')!r} — {req['description']}")
                else:
                    print(f"content_requests_node: declined for {item.get('sub_topic')!r}. "
                          f"Raw response: {response!r}")

        with open(os.path.join(state["outline_dir"], "outline_draft.json"), "w", encoding="utf-8") as f:
            json.dump(draft_outline, f, indent=2)

        # Output from this node goes to review_outline_node (the interrupt).
        return {"draft_outline": draft_outline}

    return content_requests_node
