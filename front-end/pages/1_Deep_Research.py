from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

import streamlit as st

# Make the front-end/ package root importable regardless of how Streamlit
# resolved this page's working directory.
_FRONTEND_ROOT = Path(__file__).resolve().parent.parent
if str(_FRONTEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_ROOT))

from components.layout import apply_page_config, render_footer, render_header  # noqa: E402
from services.graph_runner import (  # noqa: E402
    DEFAULT_SOURCE_BLACKLIST,
    MEDIA_ACCEPTED_EXTENSIONS,
    MEDIA_NOTES,
    MEDIA_TYPES,
    OUTPUT_FORMATS,
    REPORT_FORMATS,
    TITLE_MAX_LEN,
    TITLE_MIN_LEN,
    apply_media_requests,
    download_bundle,
    start_outline_generation,
    start_report_generation,
    sync_media_rows,
)

apply_page_config("Deep Research")
render_header("Deep Research")

# ---------------------------------------------------------------- state ---
defaults = {
    "dr_stage": "config",          # config -> outline -> generating -> done
    "dr_run": {},                  # app / thread / workspace / draft_outline
    "dr_outline_text": "",
    "dr_media_rows": [],           # [{id, theme, sub_topic, media_type, title, uploaded_file, uploaded_name}]
    "dr_gen_queue": None,
    "dr_result": None,
    "dr_error": None,
    "dr_outline_upload_error": None,
    "dr_outline_frozen_text": "",  # snapshot shown once Section 2 locks — see note at Submit Outline
    "dr_source_whitelist": [],
    "dr_source_blacklist": list(DEFAULT_SOURCE_BLACKLIST),
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


def _reset_all():
    for key, value in defaults.items():
        st.session_state[key] = value
    st.session_state.dr_outline_upload = None  # clear any file the uploader widget is still holding


def _on_outline_upload():
    """on_change callback for the outline-file uploader — runs BEFORE the
    script reruns, so it's safe to write dr_outline_text here (the text_area
    widget sharing that key hasn't been instantiated yet this run)."""
    uploaded = st.session_state.get("dr_outline_upload")
    if uploaded is None:
        return
    try:
        text = uploaded.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        st.session_state.dr_outline_upload_error = "File must be UTF-8 encoded text."
        return
    _, errors = _validate_outline_text(text)
    if errors:
        st.session_state.dr_outline_upload_error = "Uploaded file is invalid:\n" + "\n".join(f"- {e}" for e in errors)
        return
    st.session_state.dr_outline_upload_error = None
    st.session_state.dr_outline_text = text


def _validate_config(topic: str, report_format: str, output_label: str) -> list[str]:
    errors = []
    t = (topic or "").strip()
    if not t:
        errors.append("Title is required.")
    elif len(t) < TITLE_MIN_LEN:
        errors.append(f"Title must be at least {TITLE_MIN_LEN} characters.")
    elif len(t) > TITLE_MAX_LEN:
        errors.append(f"Title must be at most {TITLE_MAX_LEN} characters.")
    if report_format not in REPORT_FORMATS:
        errors.append("Report format is required.")
    if output_label not in OUTPUT_FORMATS:
        errors.append("Output format is required.")
    return errors


def _validate_outline_text(text: str) -> tuple[list[dict] | None, list[str]]:
    """Parses + validates the outline JSON text area. Returns (outline, [])
    on success, or (None, errors) — errors are shown verbatim to the user,
    so each one names the specific problem rather than a generic message."""
    errors: list[str] = []
    if not (text or "").strip():
        return None, ["Outline JSON cannot be empty."]
    try:
        outline = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, [f"Outline JSON is invalid: {exc}"]
    if not isinstance(outline, list):
        return None, ["Outline JSON must be a list of outline items."]
    # An empty list is valid, not just tolerated: the "Executive" report
    # format's only theme (EXECUTIVE SUMMARY) is a meta theme, so
    # get_standard_themes() returns no standard themes at all and plan_node
    # legitimately produces draft_outline=[] — build_work_queue_node already
    # handles this correctly (the meta section still gets queued). Requiring
    # non-empty here blocked Submit Outline for every Executive-format run.
    for i, item in enumerate(outline):
        if not isinstance(item, dict):
            errors.append(f"Item {i + 1} is not a JSON object.")
            continue
        if not str(item.get("theme") or "").strip():
            errors.append(f"Item {i + 1} is missing a non-empty 'theme'.")
        if not str(item.get("sub_topic") or "").strip():
            errors.append(f"Item {i + 1} is missing a non-empty 'sub_topic'.")
    return (outline, []) if not errors else (None, errors)


def _sections_map(outline: list[dict]) -> dict[str, list[str]]:
    """theme -> [sub_topics], in outline order, for the media table's Section
    / Sub-topic dropdowns."""
    sections: dict[str, list[str]] = {}
    for item in outline:
        theme, sub_topic = item.get("theme"), item.get("sub_topic")
        if not theme or not sub_topic:
            continue
        sections.setdefault(theme, [])
        if sub_topic not in sections[theme]:
            sections[theme].append(sub_topic)
    return sections


# ============================================================ SECTION 1 ===
st.subheader("1. Generate Outline")
section1_frozen = st.session_state.dr_stage != "config"

with st.container(border=True):
    topic = st.text_input(
        "Report title / topic", key="dr_topic", disabled=section1_frozen,
        placeholder="e.g. Cheese manufacturing in Italy", max_chars=TITLE_MAX_LEN,
    )
    if not section1_frozen:
        st.caption(f"{len(topic)}/{TITLE_MAX_LEN} characters (minimum {TITLE_MIN_LEN})")
    col_a, col_b = st.columns(2)
    with col_a:
        report_format = st.selectbox(
            "Report format", REPORT_FORMATS, key="dr_report_format",
            disabled=section1_frozen, format_func=str.title,
        )
    with col_b:
        output_label = st.selectbox(
            "Output format", list(OUTPUT_FORMATS.keys()), key="dr_output_label",
            disabled=section1_frozen,
        )

    st.multiselect(
        "Source whitelist (domains, optional)",
        options=st.session_state.dr_source_whitelist,
        key="dr_source_whitelist", accept_new_options=True, disabled=section1_frozen,
        placeholder="Type a domain and press Enter — leave empty to search everywhere",
        help="If set, web search is restricted to ONLY these domains, e.g. 'nytimes.com'.",
    )
    st.multiselect(
        "Source blacklist (domains)",
        options=st.session_state.dr_source_blacklist,
        key="dr_source_blacklist", accept_new_options=True, disabled=section1_frozen,
        placeholder="Type a domain and press Enter",
        help="These domains are excluded from web search results.",
    )

    if section1_frozen:
        st.caption(f"Locked in: **{st.session_state.dr_topic}** · "
                    f"{st.session_state.dr_report_format.title()} · {st.session_state.dr_output_label}")
        if st.session_state.dr_source_whitelist:
            st.caption(f"Whitelist: {', '.join(st.session_state.dr_source_whitelist)}")
        if st.session_state.dr_source_blacklist:
            st.caption(f"Blacklist: {', '.join(st.session_state.dr_source_blacklist)}")
    else:
        if st.button("Generate Outline", type="primary"):
            errors = _validate_config(topic, report_format, output_label)
            if errors:
                st.error("\n".join(f"- {e}" for e in errors))
            else:
                with st.spinner("Reading reference material and drafting the outline..."):
                    try:
                        run = start_outline_generation(
                            topic.strip(), report_format, OUTPUT_FORMATS[output_label],
                            source_whitelist=st.session_state.dr_source_whitelist,
                            source_blacklist=st.session_state.dr_source_blacklist,
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Outline generation failed: {exc}")
                        run = None
                if run is not None:
                    st.session_state.dr_run = run
                    st.session_state.dr_outline_text = json.dumps(run["draft_outline"], indent=2)
                    st.session_state.dr_stage = "outline"
                    st.rerun()

# ============================================================ SECTION 2 ===
if st.session_state.dr_stage in ("outline", "generating", "done"):
    st.subheader("2. Validate Outline")
    section2_frozen = st.session_state.dr_stage != "outline"

    with st.container(border=True):
        if section2_frozen:
            st.caption("Outline submitted — locked for this run.")
            st.code(st.session_state.dr_outline_frozen_text, language="json")
        else:
            st.text_area(
                "Outline (JSON) — edit sub_topics / queries / content_requests / weightage under "
                "the existing themes",
                key="dr_outline_text", height=350,
            )
            col_dl, col_ul = st.columns(2)
            with col_dl:
                st.download_button(
                    "Export outline as .json",
                    data=st.session_state.dr_outline_text.encode("utf-8"),
                    file_name="outline.json", mime="application/json",
                )
            with col_ul:
                st.file_uploader(
                    "Or upload an edited outline.json", type=["json"], key="dr_outline_upload",
                    on_change=_on_outline_upload,
                    help="Replaces the text area above with this file's content.",
                )
            if st.session_state.dr_outline_upload_error:
                st.error(st.session_state.dr_outline_upload_error)

            st.markdown("##### Supply your own media (optional)")
            # Streamlit reruns the script when the text area loses focus, so
            # this re-validates and re-syncs the media table "on blur" without
            # any extra callback wiring — see sync_media_rows' docstring for
            # exactly what gets added/removed/kept.
            outline_preview, outline_errors = _validate_outline_text(st.session_state.dr_outline_text)
            if outline_errors:
                st.warning("Fix the outline JSON above before editing media — the table below still "
                            "reflects the last valid version.\n" + "\n".join(f"- {e}" for e in outline_errors))
                sections = {}  # each row still shows its own last-known Section/Sub-topic (see below)
            else:
                st.session_state.dr_media_rows = sync_media_rows(outline_preview, st.session_state.dr_media_rows)
                sections = _sections_map(outline_preview)

            header_cols = st.columns([2, 3, 2, 3, 3, 1])
            for col, label in zip(header_cols,
                                    ["Section", "Sub-topic", "Media Type", "Title / Description", "File", ""]):
                col.markdown(f"**{label}**")

            rows_to_remove = []
            for row in st.session_state.dr_media_rows:
                rid = row["id"]
                c1, c2, c3, c4, c5, c6 = st.columns([2, 3, 2, 3, 3, 1])
                with c1:
                    # Always render (never skip the call) and always include the
                    # row's current value in `options` — a keyed selectbox whose
                    # stored value isn't among its options on a later rerun is a
                    # Streamlit crash risk, not just a display glitch.
                    theme_options = list(sections.keys())
                    current_theme = row.get("theme")
                    display_themes = theme_options[:]
                    if current_theme and current_theme not in display_themes:
                        display_themes.append(current_theme)
                    if not display_themes:
                        display_themes = ["(no sections)"]
                        current_theme = display_themes[0]
                    t_idx = display_themes.index(current_theme) if current_theme in display_themes else 0
                    selected_theme = st.selectbox(
                        "Section", display_themes, index=t_idx,
                        key=f"theme_{rid}", label_visibility="collapsed", disabled=not theme_options,
                        help="Which report section (theme) this media belongs to.",
                    )
                    if theme_options:  # only commit the selection once the outline is valid
                        row["theme"] = selected_theme
                with c2:
                    sub_options = sections.get(row.get("theme"), []) if theme_options else []
                    current_sub = row.get("sub_topic")
                    display_subs = sub_options[:]
                    if current_sub and current_sub not in display_subs:
                        display_subs.append(current_sub)
                    if not display_subs:
                        display_subs = ["(no sub-topics)"]
                        current_sub = display_subs[0]
                    s_idx = display_subs.index(current_sub) if current_sub in display_subs else 0
                    selected_sub = st.selectbox(
                        "Sub-topic", display_subs, index=s_idx,
                        key=f"subtopic_{rid}", label_visibility="collapsed", disabled=not sub_options,
                    )
                    if sub_options:
                        row["sub_topic"] = selected_sub
                with c3:
                    labels = list(MEDIA_TYPES.keys())
                    midx = labels.index(row["media_label"]) if row.get("media_label") in labels else 0
                    row["media_label"] = st.selectbox(
                        "Media type", labels, index=midx,
                        key=f"mediatype_{rid}", label_visibility="collapsed",
                    )
                    row["media_type"] = MEDIA_TYPES[row["media_label"]]
                with c4:
                    row["title"] = st.text_input(
                        "Title", value=row.get("title", ""), key=f"title_{rid}",
                        label_visibility="collapsed", placeholder="What this shows",
                    )
                with c5:
                    # Keyed per media_type: switching a row's Media Type must not
                    # leave a stale, now-mismatched file attached (e.g. a photo
                    # left wired up as a "data-table" request, which
                    # non_text_master_node would then try to open as UTF-8 text
                    # and crash on). A fresh key per type means each type keeps
                    # its own upload slot, and the row always reflects the file
                    # for whichever type is CURRENTLY selected.
                    upload_label = "Re-upload file" if row.get("uploaded_name") else "Upload file"
                    uploaded = st.file_uploader(
                        upload_label, key=f"upload_{rid}_{row['media_type']}", label_visibility="visible",
                        type=MEDIA_ACCEPTED_EXTENSIONS[row["media_type"]],
                        help=MEDIA_NOTES[row["media_type"]],
                    )
                    row["uploaded_file"] = uploaded
                    row["uploaded_name"] = uploaded.name if uploaded is not None else None
                with c6:
                    st.write("")
                    if st.button("✕", key=f"remove_{rid}", help="Remove row"):
                        rows_to_remove.append(rid)

            if rows_to_remove:
                st.session_state.dr_media_rows = [
                    r for r in st.session_state.dr_media_rows if r["id"] not in rows_to_remove
                ]
                st.rerun()

            if st.button("+ Add media row", disabled=not sections):
                first_theme = next(iter(sections), None)
                st.session_state.dr_media_rows.append({
                    "id": uuid.uuid4().hex,
                    "theme": first_theme,
                    "sub_topic": sections[first_theme][0] if first_theme else None,
                    "media_label": list(MEDIA_TYPES.keys())[0],
                    "media_type": list(MEDIA_TYPES.values())[0],
                    "title": "",
                    "source": "manual",
                    "orig_description": None,
                    "uploaded_file": None,
                    "uploaded_name": None,
                })
                st.rerun()

            st.divider()
            if st.button("Submit Outline", type="primary"):
                outline, errors = _validate_outline_text(st.session_state.dr_outline_text)
                if errors:
                    st.error("\n".join(f"- {e}" for e in errors))
                else:
                    run = st.session_state.dr_run
                    try:
                        manifest = apply_media_requests(outline, run["workspace"]["outline_dir"],
                                                          st.session_state.dr_media_rows)
                    except Exception as exc:  # noqa: BLE001 - e.g. a malformed CSV/table upload
                        st.error(f"Couldn't attach a media request: {exc}")
                        manifest = None
                    if manifest is not None:
                        queue_box = start_report_generation(run["app"], run["thread"], outline, manifest)
                        st.session_state.dr_gen_queue = queue_box
                        # NOT dr_outline_text: that key's widget (the text_area
                        # above) already rendered earlier in this same script
                        # run, and Streamlit raises StreamlitWidgetAlreadyInstantiatedError
                        # if a bound key is written after its widget has run.
                        # dr_outline_frozen_text is a plain, unbound key used
                        # only for the locked-section display below.
                        st.session_state.dr_outline_frozen_text = json.dumps(outline, indent=2)
                        st.session_state.dr_stage = "generating"
                        st.rerun()

    if st.session_state.dr_stage == "generating":
        st.info("Researching, writing, and validating each section — this can take a few minutes.")
        with st.spinner("Generating report..."):
            q = st.session_state.dr_gen_queue
            try:
                status, payload = q.get(timeout=2)
            except Exception:  # queue.Empty
                status, payload = None, None
        if status == "ok":
            st.session_state.dr_result = payload
            st.session_state.dr_stage = "done"
            st.rerun()
        elif status == "error":
            st.session_state.dr_error = payload
            st.session_state.dr_stage = "outline"
            st.rerun()
        else:
            time.sleep(1)
            st.rerun()

if st.session_state.dr_error:
    st.error(f"Report generation failed: {st.session_state.dr_error}")
    st.session_state.dr_error = None

# ============================================================ SECTION 3 ===
if st.session_state.dr_stage == "done" and st.session_state.dr_result:
    st.subheader("3. Report")
    result = st.session_state.dr_result
    with st.container(border=True):
        st.success("Report generated.")
        bundle = download_bundle(result)
        st.download_button(
            "Download report", data=bundle["data"],
            file_name=bundle["filename"], mime=bundle["mime"], type="primary",
        )
        with st.expander("Preview", expanded=True):
            if result.get("output_format") in ("markdown", None):
                # The markdown this report builds deliberately embeds raw
                # <a id="..."> anchors (TOC/heading/citation links — see
                # nodes_output.py's heading-block comment) for Pandoc's
                # CommonMark reader to resolve on DOCX/PDF export. Streamlit's
                # own markdown renderer escapes raw HTML by default, which
                # left those literal <a ...></a> tags visible as text in this
                # preview instead of being invisible anchors.
                st.markdown(result.get("final_report", ""), unsafe_allow_html=True)
            else:
                st.code(result.get("final_report", ""), language=result.get("output_format"))

        if st.button("Start a new report"):
            _reset_all()
            st.rerun()

render_footer()
