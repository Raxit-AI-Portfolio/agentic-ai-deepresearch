"""Bridges the Streamlit UI to agentic_core's LangGraph pipeline.

Pages never touch the graph, state.py's ResearchState, or the checkpointer
directly — they call the functions here and read/write plain
st.session_state values. This keeps the LangGraph wiring (and its
interrupt/resume contract, see agentic_core/nodes_plan.py) in one place.
"""

from __future__ import annotations

import copy
import io
import os
import queue
import re
import sys
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# agentic_core lives at the repo root, one level above front-end/.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from langgraph.types import Command  # noqa: E402

from agentic_core.config import ReportConfig, ResearchConfig  # noqa: E402
from agentic_core.graph import build_graph  # noqa: E402
from agentic_core.nodes_plan import add_content_request  # noqa: E402
from agentic_core.reference_docs import create_run_workspace  # noqa: E402

REPORT_FORMATS = ["executive", "detailed", "academic", "presentation"]

# Pre-populated in the Source blacklist control (Section 1) — sites that are
# rarely useful as citable research sources for this kind of report. The
# user can still remove any of these or add more.
DEFAULT_SOURCE_BLACKLIST = [
    "pinterest.com", "facebook.com", "instagram.com",
    "youtube.com", "twitter.com", "reddit.com",
]

OUTPUT_FORMATS = {
    "Markdown": "markdown",
    "HTML": "html",
    "JSON": "json",
    "XML": "xml",
    "DOCX": "docx",
    "PDF": "pdf",
}

MEDIA_TYPES = {
    "Photo": "internet-image",
    "Data Table": "data-table",
    "Diagram": "diagram",
}

# non_text_master_node (agentic_core/nodes_research.py) reads a user-supplied
# "internet-image" file as a binary image, and a "data-table"/"diagram" file
# as raw UTF-8 text (markdown-table text / Mermaid syntax, respectively) — it
# never parses CSV. These extension lists and notes match that contract; the
# CSV convenience for data tables is handled in save_uploaded_media below.
MEDIA_ACCEPTED_EXTENSIONS = {
    "internet-image": ["png", "jpg", "jpeg", "webp", "gif"],
    "data-table": ["csv", "md", "txt"],
    "diagram": ["mmd", "md", "txt"],
}
MEDIA_NOTES = {
    "internet-image": "Upload an image file (PNG, JPG, JPEG, WEBP, or GIF) to use as the section's photo.",
    "data-table": "Upload a CSV file, or a Markdown-formatted table (.md/.txt). "
                  "CSV is automatically converted to a Markdown table.",
    "diagram": "Upload a Mermaid diagram definition (.mmd, .md, or .txt) — raw Mermaid syntax, "
               "e.g. 'flowchart TD\\n    A --> B'.",
}

TITLE_MIN_LEN = 3
TITLE_MAX_LEN = 200

_EXPORT_MIME = {
    "html": "text/html",
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
_TEXT_MIME = {"markdown": "text/markdown", "json": "application/json", "xml": "application/xml"}
_TEXT_EXT = {"markdown": "md", "json": "json", "xml": "xml"}


@st.cache_resource(show_spinner=False)
def _get_graph_app():
    """Built once per server process (loads the LLM, search tool, vision
    checker) and reused across every report run — the checkpointer inside
    it is keyed per-run by thread_id, so concurrent runs don't collide."""
    return build_graph(ResearchConfig(), ReportConfig())


def _thread_config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def normalize_domains(values: List[str]) -> List[str]:
    """Cleans user-typed whitelist/blacklist entries into the bare-hostname
    form nodes_research.py expects — it uses each entry directly in a
    `site:<domain>`/`-site:<domain>` search operator and substring-matches
    it against a result URL's netloc (`agentic_core/nodes_research.py`
    research_node), so a scheme prefix or trailing path would silently
    never match anything. Strips "http(s)://", any path/query after the
    host, surrounding whitespace, lowercases, and de-dupes (order
    preserved) while dropping anything that's empty after cleaning."""
    cleaned: List[str] = []
    for raw in values:
        d = (raw or "").strip().lower()
        d = re.sub(r"^https?://", "", d)
        d = d.split("/", 1)[0]
        if d and d not in cleaned:
            cleaned.append(d)
    return cleaned


def start_outline_generation(
    topic: str, report_format: str, output_format: str,
    source_whitelist: Optional[List[str]] = None,
    source_blacklist: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Runs the graph from START through the review_outline interrupt.

    Synchronous by design: this leg is reference-doc parsing + one LLM call
    per theme, not the long research/write/validate loop, so a blocking
    spinner is appropriate here.

    source_whitelist/source_blacklist aren't used until research_node's web
    search (after review_outline resumes) — reference_docs/plan don't search
    the web — but they're set here, at the initial invoke, because that's
    the only state LangGraph's checkpointer carries across the interrupt:
    Command(resume=...) only ever supplies {"outline", "user_provided_content"}
    (see nodes_plan.py's resume contract), so anything else the research
    loop needs has to already be in state before the graph pauses.
    """
    app = _get_graph_app()
    workspace = create_run_workspace(reference_zip_path=None)
    thread = _thread_config(workspace["run_id"])
    initial_input = {
        **workspace,
        "topic": topic,
        "report_format": report_format,
        "output_format": output_format,
        "source_whitelist": normalize_domains(source_whitelist or []),
        "source_blacklist": normalize_domains(source_blacklist or []),
    }
    result = app.invoke(initial_input, thread)
    interrupt_payload = result["__interrupt__"][0].value
    return {
        "app": app,
        "thread": thread,
        "workspace": workspace,
        "draft_outline": interrupt_payload["draft_outline"],
    }


def save_uploaded_media(outline_dir: str, uploaded_file, media_type: Optional[str] = None) -> str:
    """Saves a Streamlit UploadedFile into <outline_dir>/contents/ (the
    location build_work_queue_node expects, per nodes_plan.py's resume
    contract) and returns the filename actually used on disk.

    A "data-table" upload that's a .csv is converted to an actual Markdown
    table here, since non_text_master_node reads a user-supplied data-table
    file back as raw text markup, not parsed CSV — agentic_core itself is
    left untouched, this only changes what bytes land on disk before the
    graph ever sees the file. The dedup loop below runs against the FINAL
    on-disk extension (.md for a converted CSV, unchanged otherwise) so a
    csv->md conversion can never silently overwrite an unrelated existing
    .md file that happens to share the CSV's base name.
    """
    contents_dir = os.path.join(outline_dir, "contents")
    os.makedirs(contents_dir, exist_ok=True)
    base, ext = os.path.splitext(uploaded_file.name)
    convert_csv = media_type == "data-table" and ext.lower() == ".csv"
    final_ext = ".md" if convert_csv else ext

    name = f"{base}{final_ext}"
    path = os.path.join(contents_dir, name)
    n = 1
    while os.path.exists(path):
        name = f"{base}_{n}{final_ext}"
        path = os.path.join(contents_dir, name)
        n += 1

    if convert_csv:
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        with open(path, "w", encoding="utf-8") as f:
            f.write(df.to_markdown(index=False))
    else:
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return name


def _find_content_request(item: dict, req_type: str, description: str) -> Optional[dict]:
    for req in item.get("content_requests") or []:
        if req.get("type") == req_type and (req.get("description") or "") == description:
            return req
    return None


def sync_media_rows(outline: List[dict], rows: List[dict]) -> List[dict]:
    """Keeps the media-request table in sync with the outline JSON text area,
    called on every render of Section 2 (Streamlit already reruns the script
    when the text area loses focus, so this effectively runs "on blur"):

    - a row whose sub_topic no longer exists in the outline is dropped
    - a "json" row (one that mirrors a content_request already written into
      the outline JSON) is dropped if that specific content_request was
      removed, even if the sub_topic itself remains
    - a row's Section (theme) label is refreshed if the item moved themes
    - a fresh "json" row is added for any content_request present in the
      JSON that isn't reflected in the table yet (e.g. the user hand-typed
      one, or a theme/sub-topic carrying one was just pasted in)
    - "manual" rows (added via the UI's "+ Add media row", not yet written
      into the JSON) are left alone as long as their sub_topic still exists

    Rows that survive are kept BY IDENTITY (the same dict, same id) rather
    than rebuilt, so an uploaded file and its "Re-upload" label survive a
    sync untouched — only genuinely new/removed rows change.

    A "json" row is matched back to its content_request by (sub_topic, type,
    description) — content_requests carry no id of their own. Editing that
    request's description text directly in the JSON (rather than through the
    table's Title field) will therefore read as delete-old/add-new and drop
    any upload already attached to it; editing via the Title field does not
    have this problem, since the row's original-description key is kept
    separately from the editable title.
    """
    by_subtopic = {it.get("sub_topic"): it for it in outline if isinstance(it, dict) and it.get("sub_topic")}
    label_by_type = {v: k for k, v in MEDIA_TYPES.items()}

    kept: List[dict] = []
    seen_json_keys = set()
    for row in rows:
        item = by_subtopic.get(row.get("sub_topic"))
        if item is None:
            continue
        if row.get("source") == "json":
            req = _find_content_request(item, row.get("media_type"), row.get("orig_description", ""))
            if req is None:
                continue
            seen_json_keys.add((row["sub_topic"], row["media_type"], row.get("orig_description", "")))
        row["theme"] = item.get("theme")
        kept.append(row)

    for item in outline:
        sub_topic = item.get("sub_topic")
        if not sub_topic:
            continue
        for req in item.get("content_requests") or []:
            rtype = req.get("type")
            if rtype not in label_by_type:
                continue
            rdesc = req.get("description") or ""
            key = (sub_topic, rtype, rdesc)
            if key in seen_json_keys:
                continue
            kept.append({
                "id": uuid.uuid4().hex, "theme": item.get("theme"), "sub_topic": sub_topic,
                "media_label": label_by_type[rtype], "media_type": rtype, "title": rdesc,
                "orig_description": rdesc, "source": "json",
                "uploaded_file": None, "uploaded_name": None,
            })
            seen_json_keys.add(key)

    return kept


def apply_media_requests(outline: List[dict], outline_dir: str, media_rows: List[dict]) -> Dict[str, dict]:
    """Attaches each media-table row to its sub_topic's content_requests and,
    for rows that carry an upload, saves the file and returns the
    user_provided_content manifest for Command(resume=...).

    A "json" row (see sync_media_rows) already has a matching content_request
    in the outline — it's updated in place (description from the row's Title,
    user_supplied_path if a file was uploaded) rather than duplicated. A
    "manual" row (added via "+ Add media row", not yet in the JSON) gets a
    brand new content_request via add_content_request. Rows missing a
    sub_topic or media type are skipped.
    """
    manifest: Dict[str, dict] = {}
    by_subtopic = {it.get("sub_topic"): it for it in outline if isinstance(it, dict) and it.get("sub_topic")}
    for row in media_rows:
        sub_topic = row.get("sub_topic")
        media_type = row.get("media_type")
        if not sub_topic or not media_type:
            continue
        filename: Optional[str] = None
        if row.get("uploaded_file") is not None:
            filename = save_uploaded_media(outline_dir, row["uploaded_file"], media_type)
        path = os.path.join(outline_dir, "contents", filename) if filename else None

        if row.get("source") == "json":
            item = by_subtopic.get(sub_topic)
            req = item and _find_content_request(item, media_type, row.get("orig_description", ""))
            if req is None:
                continue  # stale — sync_media_rows should have already dropped this row
            if row.get("title"):
                req["description"] = row["title"]
            if path:
                req["user_supplied_path"] = path
                manifest[filename] = {"theme": item.get("theme"), "sub_topic": sub_topic, "type": media_type}
            continue

        item = add_content_request(
            outline, sub_topic, media_type,
            description=row.get("title") or sub_topic, user_supplied_path=path,
        )
        if filename:
            manifest[filename] = {"theme": item.get("theme"), "sub_topic": sub_topic, "type": media_type}
    return manifest


def _run_in_thread(app, thread, outline, manifest, out_box: "queue.Queue"):
    try:
        result = app.invoke(Command(resume={"outline": outline, "user_provided_content": manifest}), thread)
        out_box.put(("ok", result))
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
        out_box.put(("error", str(exc)))


def start_report_generation(app, thread, outline: List[dict], manifest: Dict[str, dict]) -> "queue.Queue":
    """Resumes the graph past review_outline in a background thread — this
    leg runs the full research/write/validate loop and can take minutes."""
    out_box: "queue.Queue" = queue.Queue()
    t = threading.Thread(
        target=_run_in_thread,
        args=(app, thread, copy.deepcopy(outline), manifest, out_box),
        daemon=True,
    )
    t.start()
    return out_box


def _collect_media_files(result: Dict[str, Any]) -> List[Dict[str, str]]:
    """Unique on-disk media files referenced by any section's media list.

    content.py's internet_image_content/render_chart save the actual bytes
    to the OS temp dir (ContentResult["path"]) — data-table/diagram results
    carry no file, just inline "markup" text, so there's nothing to collect
    for those. A rejected/failed item (see format_sections_node's
    is_placeholder check) also has no path, so it's naturally skipped here.
    """
    seen: Dict[str, str] = {}
    for sec in result.get("sections", []) or []:
        for m in sec.get("media", []) or []:
            path = m.get("path")
            if path and os.path.exists(path) and path not in seen:
                seen[path] = f"media/{os.path.basename(path)}"
    return [{"path": p, "arcname": arcname} for p, arcname in seen.items()]


def _rewrite_media_refs(text: str, media_files: List[Dict[str, str]]) -> str:
    """Points the report's image references at the zip-relative "media/..."
    paths instead of the original absolute temp-file path.

    format_sections_node (agentic_core/nodes_output.py) embeds media two
    ways depending on output format: html/markdown use the path lstripped of
    a leading "/" as `<img src="...">` (an absolute local filesystem path,
    only ever valid on the machine that generated the report), while
    xml/json use just `os.path.basename(path)`. Each file's two possible
    on-disk references are swapped for a private token before the token is
    replaced with the real arcname, so replacing the (basename-only)
    reference first can't accidentally clobber part of the (path-prefixed)
    reference for the same file, or vice versa.
    """
    for mf in media_files:
        token = f"\x00MEDIA_TOKEN_{uuid.uuid4().hex}\x00"
        abs_ref = mf["path"].lstrip("/")
        base_ref = os.path.basename(mf["path"])
        text = text.replace(abs_ref, token).replace(base_ref, token)
        text = text.replace(token, mf["arcname"])
    return text


def download_bundle(result: Dict[str, Any]) -> Dict[str, Any]:
    """Builds {data, filename, mime} for the Report section's download
    button — covers both export_path formats (html/pdf/docx, or their
    markdown fallback when the optional system libs aren't installed) and
    the in-memory-string formats (markdown/json/xml).

    When the report has any generated/uploaded media with an actual file on
    disk (photos, rendered charts — see _collect_media_files), the download
    becomes a report.zip containing the report plus a media/ folder, instead
    of just the bare report file, so the report is still viewable once it's
    unzipped somewhere else. PDF/DOCX exports already have their images
    baked into the binary at render time (weasyprint/pypandoc read the local
    file path while it still exists) so their report text isn't rewritten —
    the media/ folder in that case is just the original files as a bonus,
    not something the exported file depends on.
    """
    fmt = result.get("output_format") or "markdown"
    export_path = result.get("export_path")
    media_files = _collect_media_files(result)

    if export_path and os.path.exists(export_path):
        ext = os.path.splitext(export_path)[1].lstrip(".")
        with open(export_path, "rb") as f:
            report_bytes = f.read()
        report_filename = f"report.{ext}"
        mime = _EXPORT_MIME.get(ext, "text/markdown")
        rewrite = ext in ("html", "md")
    else:
        ext = _TEXT_EXT.get(fmt, "txt")
        report_bytes = (result.get("final_report") or "").encode("utf-8")
        report_filename = f"report.{ext}"
        mime = _TEXT_MIME.get(fmt, "text/plain")
        rewrite = True

    if not media_files:
        return {"data": report_bytes, "filename": report_filename, "mime": mime}

    if rewrite:
        report_bytes = _rewrite_media_refs(report_bytes.decode("utf-8"), media_files).encode("utf-8")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(report_filename, report_bytes)
        for mf in media_files:
            zf.write(mf["path"], mf["arcname"])
    return {"data": buf.getvalue(), "filename": "report.zip", "mime": "application/zip"}
