"""Nodes — citations, formatting, assembly, export, cleanup."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import date

from .services import _linkify_urls
from .state import ResearchState, SourceRef

# Display width in pixels — separate from content.py's MAX_IMAGE_DIMENSION
# (which caps the actual file's resolution/size on disk). This controls how
# big the image LOOKS when rendered, regardless of the underlying file.
DISPLAY_IMAGE_WIDTH = 480


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def _unique_slug(text: str, seen: set) -> str:
    """De-dupes anchors across the report (e.g. two sections that both ended
    up titled "Sustainable Cheese Production") so TOC links don't collide."""
    base = _slugify(text)
    slug, i = base, 2
    while slug in seen:
        slug = f"{base}-{i}"
        i += 1
    seen.add(slug)
    return slug


def _linkify_citation_numbers(text: str, fmt: str, citation_count: int) -> str:
    """Turns an inline "[N]" citation marker (written by write_node per its
    prompt instructions) into a link down to that citation's entry in the
    Citations list — see the "cite-{num}" anchors in _cite_line_md and
    _cite_line_html. Only linkifies numbers within the actual citation range
    (1..citation_count), so an incidental bracketed number in the model's own
    prose (a year, an unrelated footnote-style bracket) is left alone."""
    def repl(m):
        num = int(m.group(1))
        if not (1 <= num <= citation_count):
            return m.group(0)
        if fmt == "html":
            return f'<a href="#cite-{num}">[{num}]</a>'
        if fmt in ("markdown", "pdf", "docx"):
            return f"[[{num}]](#cite-{num})"
        return m.group(0)
    return re.sub(r"\[(\d+)\]", repl, text)


def assign_citations_node(state: ResearchState) -> dict:
    """citation_map (url -> global citation number) and citations (the
    ordered SourceRef list) are now built incrementally by research_node, as
    each source is first discovered — not rebuilt here from state["sections"]
    in section order — because write_node shows the model that same running
    citation_map so it can cite sources inline as "[N]" while writing. This
    node just passes through the final accumulated result, with one filter:
    a web source that research_node fetched but no section ever actually
    cited inline as "[N]" is dropped from the final list — filter_relevant_
    results (services.py) reduces but doesn't eliminate this (an unrelated
    search result scoring just above its relevance threshold, then never
    used in any section's prose — observed directly in a car-market report,
    see open-issues.md); this is the second, exact backstop: if a source was
    never actually used, it shouldn't appear in Citations regardless of why.
    Reference docs are kept unconditionally per the architectural rule that
    they always get a citation entry even though they're exempt from the
    copyright-overlap check."""
    citation_map = state.get("citation_map", {})
    citations = state.get("citations", [])
    cited_numbers = set()
    for sec in state.get("sections", []):
        cited_numbers.update(int(n) for n in re.findall(r"\[(\d+)\]", sec.get("content", "") or ""))
    used_citations = [
        c for c in citations
        if c.get("kind") == "reference" or citation_map.get(c.get("url")) in cited_numbers
    ]
    # Output from this node goes to the make_format_sections_node() to format each section's
    # content + media into the requested output format (markdown, HTML, or XML). The formatted sections are then passed to assemble_report_node for final assembly into a single report string.
    return {"citation_map": citation_map, "citations": used_citations}


def make_format_sections_node():
    def format_sections_node(state: ResearchState) -> dict:
        """Formats each section's content + media into the requested output format (markdown, HTML, or XML). The formatted sections are then passed to
        assemble_report_node for final assembly into a single report string."""
        fmt = state["output_format"]
        citation_map = state.get("citation_map", {})
        formatted = []
        seen_anchors = set()
        for sec in state["sections"]:
            display_title = sec.get("sub_topic") if sec["kind"] == "standard" else sec["theme"].title()
            anchor = _unique_slug(display_title, seen_anchors)
            body = _linkify_urls(sec["content"], citation_map)
            # A section with no sources of its own has nothing valid to cite
            # — pass citation_count=0 so any stray "[N]" the model wrote
            # anyway (write_node's prompt now tries to prevent this, but a
            # small model doesn't always follow instructions) is left as
            # plain text instead of being linkified to a citation from some
            # OTHER section's research (citation_map accumulates globally).
            citation_count = len(citation_map) if sec.get("sources") else 0
            body = _linkify_citation_numbers(body, fmt, citation_count)
            media_markup = []
            media_json = []
            for m in sec.get("media", []):
                # A failed/rejected media item never becomes a rendered asset
                # in ANY format — make_diagram_content sets status "rejected"
                # when the LLM's Mermaid output doesn't validate, and
                # internet_image_content/data_table_content leave both "path"
                # and "markup" unset when a fetch/parse fails. Every format
                # (including JSON/XML) gets the same placeholder marker
                # instead of silently embedding broken content or, for JSON,
                # dumping the raw failed ContentResult (with its
                # validation_notes) as if it were real report content.
                placeholder_note = f"content-type: {m['type']} | description: {m['caption']}"
                is_placeholder = m.get("status") == "rejected" or not (m.get("path") or m.get("markup"))

                if is_placeholder:
                    if fmt == "json":
                        media_json.append({"comment": f"TODO: {placeholder_note}"})
                    else:
                        media_markup.append(f"<!-- TODO: {placeholder_note} -->")
                    continue

                if m.get("path"):
                    # The file itself lives at an absolute path like
                    # "/tmp/img_....jpg" — correct for actual Python file I/O
                    # (cleanup_node, copying to Drive, etc.), but a leading
                    # "/" in an <img src="..."> is a ROOT-RELATIVE URL to
                    # whatever is rendering the report, not a local
                    # filesystem path, so it never resolves there. Strip it
                    # only for this display reference.
                    display_path = m["path"].lstrip("/")
                    if fmt == "html":
                        media_markup.append(
                            f'<figure><img src="{display_path}" width="{DISPLAY_IMAGE_WIDTH}">'
                            f'<figcaption>{m["caption"]}</figcaption></figure>'
                        )
                    elif fmt == "xml":
                        media_markup.append(f'<media path="{os.path.basename(m["path"])}" caption="{m["caption"]}"/>')
                    elif fmt == "json":
                        media_json.append({"path": os.path.basename(m["path"]), "caption": m["caption"]})
                    else:
                        # Bare markdown ![]() has no way to constrain display
                        # size — it renders at the image's native resolution.
                        # An inline <img width="..."> (markdown renderers,
                        # including Colab's, pass raw HTML through) gives
                        # actual control over how big it appears, independent
                        # of the underlying file's real pixel dimensions.
                        media_markup.append(
                            f'<img src="{display_path}" alt="{m["caption"]}" width="{DISPLAY_IMAGE_WIDTH}">\n\n*{m["caption"]}*'
                        )
                else:  # m.get("markup")
                    if fmt == "html":
                        media_markup.append(f'<pre>{m["markup"]}</pre><p>{m["caption"]}</p>')
                    elif fmt == "xml":
                        media_markup.append(f'<media caption="{m["caption"]}"><![CDATA[{m["markup"]}]]></media>')
                    elif fmt == "json":
                        media_json.append({"markup": m["markup"], "caption": m["caption"]})
                    else:
                        if m["type"] == "data-table":
                            # A code fence would make a markdown renderer show
                            # this as literal preformatted text instead of an
                            # actual table — the markup IS markdown table
                            # syntax already, so it must stay unfenced.
                            media_markup.append(f"{m['markup']}\n\n*{m['caption']}*")
                        else:
                            fence = "mermaid" if m["type"] == "diagram" else ""
                            media_markup.append(f"```{fence}\n{m['markup']}\n```\n*{m['caption']}*")

            if fmt == "json":
                formatted.append({"title": display_title, "content": body, "sources": sec["sources"],
                                   "media": media_json})
                continue
            # Same level for every section regardless of kind (meta vs.
            # standard) — the theme a standard section belongs to is never
            # shown anywhere in the document, so giving it a smaller heading
            # than meta sections read as an unexplained, inconsistent size
            # difference rather than a visible parent/child hierarchy.
            heading_level = "h2"
            if fmt == "html":
                title = f'<{heading_level} id="{anchor}">{display_title}</{heading_level}>'
                markup = f"{title}<p>{body}</p>" + "".join(media_markup)
            elif fmt == "xml":
                markup = (f'<section title="{display_title}">'
                          f'<content><![CDATA[{body}]]></content>{"".join(media_markup)}</section>')
            else:  # markdown / pdf / docx
                # Blank lines are required here, not cosmetic — CommonMark
                # (what Pandoc's markdown reader uses) treats a line starting
                # with raw HTML like <a id="..."></a> as the start of an HTML
                # BLOCK, which swallows every following line as literal text
                # until it hits a blank line. A single "\n" between the
                # anchor and the heading below it wasn't enough — the
                # "### Heading" line was getting absorbed into that HTML
                # block instead of being parsed as a heading, so it rendered
                # as literal "### Heading" text in the exported DOCX/PDF
                # instead of an actual heading style. Needs a real blank
                # line ("\n\n") to end the HTML block first.
                heading = f'<a id="{anchor}"></a>\n\n## {display_title}\n\n'
                media_block = ("\n\n" + "\n\n".join(media_markup)) if media_markup else ""
                markup = f"{heading}{body}{media_block}\n\n"
            formatted.append({"theme": sec["theme"], "markup": markup, "title": display_title, "anchor": anchor})
        return {"formatted_sections": formatted}

    # Output from this node goes to the assemble_report_node() to combine all formatted sections into a single report string, 
    # along with the table of contents and citations section.
    return format_sections_node


def _cite_line_md(c: SourceRef, num: int) -> str:
    if c.get("kind") == "reference":
        return f'- <a id="cite-{num}"></a>[{num}] Reference: {c["title"]} (user-provided)'
    return f'- <a id="cite-{num}"></a>[{num}] [{c["title"]}]({c["url"]})'


def _cite_line_html(c: SourceRef, num: int) -> str:
    if c.get("kind") == "reference":
        return f'<li id="cite-{num}">[{num}] Reference: {c["title"]} (user-provided)</li>'
    return f'<li id="cite-{num}">[{num}] <a href="{c["url"]}">{c["title"]}</a></li>'


def _cite_line_xml(c: SourceRef, num: int) -> str:
    if c.get("kind") == "reference":
        return f'<citation ref="{num}" kind="reference">{c["title"]}</citation>'
    return f'<citation ref="{num}" kind="web" url="{c["url"]}">{c["title"]}</citation>'


def assemble_report_node(state: ResearchState) -> dict:
    """Assembles the final report string from the formatted sections, table of contents, and citations, 
    in the requested output format (markdown, HTML, or XML)."""
    fmt, topic, today = state["output_format"], state["topic"], date.today().isoformat()
    citations = state.get("citations", [])
    cmap = state.get("citation_map", {})

    def num_of(c: SourceRef) -> int:
        return cmap.get(c["url"], 0)

    if fmt == "json":
        toc = [sec.get("sub_topic") or sec["theme"].title() for sec in state["sections"]]
        report = {"title": f"Research Report: {topic}", "date": today, "version": "1.0",
                  "toc": toc, "sections": state["formatted_sections"], "citations": citations}
        return {"final_report": json.dumps(report, indent=2)}

    if fmt == "xml":
        toc = [sec.get("sub_topic") or sec["theme"].title() for sec in state["sections"]]
        toc_xml = "".join(f"<item>{t}</item>" for t in toc)
        cite_xml = "".join(_cite_line_xml(c, num_of(c)) for c in citations)
        body_xml = "".join(s["markup"] for s in state["formatted_sections"])
        return {"final_report": (
            f"<report><title>Research Report: {topic}</title><date>{today}</date>"
            f"<version>1.0</version><toc>{toc_xml}</toc>{body_xml}<citations>{cite_xml}</citations></report>"
        )}

    if fmt == "html":
        toc_html = "".join(f'<li><a href="#{s["anchor"]}">{s["title"]}</a></li>' for s in state["formatted_sections"])
        cite_html = "".join(_cite_line_html(c, num_of(c)) for c in citations)
        body_html = "".join(s["markup"] for s in state["formatted_sections"])
        return {"final_report": (
            f"<html><head><title>Research Report: {topic}</title></head><body>"
            f"<h1>Research Report: {topic}</h1><p>Date: {today} | Version: 1.0</p>"
            f"<h2>Table of Contents</h2><ul>{toc_html}</ul>{body_html}"
            f"<h2>Citations</h2><ul>{cite_html}</ul></body></html>"
        )}

    toc_md = "\n".join(f'- [{s["title"]}](#{s["anchor"]})' for s in state["formatted_sections"])
    cite_md = "\n".join(_cite_line_md(c, num_of(c)) for c in citations)
    body_md = "".join(s["markup"] for s in state["formatted_sections"])
    
    # Output from this node goes to the export_node() to write the final report to a file in the requested output format (HTML, PDF, or DOCX),
    # or to a Markdown file if the required libraries are not installed. The export path is returned in the state for display to the user.
    return {"final_report": (
        f"# Research Report: {topic}\n\n*Date: {today} | Version: 1.0*\n\n"
        f"## Table of Contents\n{toc_md}\n\n{body_md}\n## Citations\n{cite_md}\n"
    )}

def export_node(state: ResearchState) -> dict:
    """Writes the final report to a file in the requested output format (HTML, PDF, or DOCX),
    or to a Markdown file if the required libraries are not installed. The export path is returned in the state for display to the user."""
    
    fmt = state["output_format"]
    if fmt not in ("html", "pdf", "docx"):
        return {}
    slug = re.sub(r"[^a-z0-9]+", "_", state["topic"].lower()).strip("_") or "report"
    tmp_dir = tempfile.gettempdir()

    if fmt == "html":
        path = os.path.join(tmp_dir, f"{slug}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(state["final_report"])
        return {"export_path": path}

    if fmt == "pdf":
        try:
            import markdown
            from weasyprint import HTML
            path = os.path.join(tmp_dir, f"{slug}.pdf")
            HTML(string=markdown.markdown(state["final_report"])).write_pdf(path)
            return {"export_path": path}
        except ImportError:
            path = os.path.join(tmp_dir, f"{slug}.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(state["final_report"])
            return {"export_path": path, "final_report": state["final_report"] +
                     "\n\n[PDF export needs `pip install markdown weasyprint` — wrote Markdown instead.]"}

    try:
        import pypandoc
        path = os.path.join(tmp_dir, f"{slug}.docx")
        pypandoc.convert_text(state["final_report"], "docx", format="md", outputfile=path)
        return {"export_path": path}
    except ImportError:
        path = os.path.join(tmp_dir, f"{slug}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(state["final_report"])
        return {"export_path": path, "final_report": state["final_report"] +
                 "\n\n[DOCX export needs `pip install pypandoc` (+ pandoc binary) — wrote Markdown instead.]"}

def cleanup_node(state: ResearchState) -> dict:
    """Deletes raw reference-doc uploads and their extracted text once the
    report is generated — we're not building a persistent knowledge base, so
    there's no reason to keep them around. outline.json and the exported
    report are left untouched; non-text media files (used by the report
    itself) are also left untouched."""
    import os
    import shutil

    outline_dir = state.get("outline_dir")
    if outline_dir:
        for sub in ("reference_docs", "reference_docs_raw"):
            path = os.path.join(outline_dir, sub)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
    return {}