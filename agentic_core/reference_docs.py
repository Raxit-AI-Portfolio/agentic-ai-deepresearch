"""
Nodes - reference_docs_node

Reference documents — extraction (no RAG: plain text in, plain text out).

Matching a section to its relevant reference doc(s) uses TF-IDF + cosine
similarity (match_reference_docs) — a basic-NLP, embedding-free alternative to
RAG that's appropriate at this scale (a handful of documents, queried once;
no persistent corpus, no index needed). Supported input types: text PDF,
scanned PDF (auto-detected per page), .docx, .html, and images with text (all
via OCR/parsing helpers, no RAG pipeline)."""

from __future__ import annotations

import os
import uuid
import zipfile
from typing import Dict, List, Optional

from .constants import DATA_DIR
from .state import ResearchState


def _ocr_image(path: str) -> str:
    import pytesseract
    from PIL import Image
    return pytesseract.image_to_string(Image.open(path))


def _ocr_pdf_page(path: str, page_index: int) -> str:
    import fitz  # PyMuPDF — pure-pip rasterization, avoids the poppler system dependency
    import pytesseract
    from PIL import Image
    import io
    doc = fitz.open(path)
    pix = doc[page_index].get_pixmap(dpi=200)
    return pytesseract.image_to_string(Image.open(io.BytesIO(pix.tobytes("png"))))


def _extract_pdf(path: str, min_chars_per_page: int = 40) -> str:
    """Text-layer extraction via pdfplumber; auto-falls-back to OCR per page
    when too little text is found — handles text PDFs and scanned PDFs with
    the same function, no manual flag needed."""
    import pdfplumber
    parts = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()
            if len(text) < min_chars_per_page:
                text = _ocr_pdf_page(path, i)
            parts.append(text)
    return "\n\n".join(parts)


def _extract_docx(path: str) -> str:
    import docx
    return "\n".join(p.text for p in docx.Document(path).paragraphs)


def _extract_html(path: str) -> str:
    from bs4 import BeautifulSoup
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return BeautifulSoup(f.read(), "html.parser").get_text(separator="\n")


def _extract_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


REFERENCE_EXTRACTORS = {
    ".pdf": _extract_pdf, ".docx": _extract_docx, ".doc": _extract_docx,
    ".html": _extract_html, ".htm": _extract_html,
    ".png": _ocr_image, ".jpg": _ocr_image, ".jpeg": _ocr_image,
    ".txt": _extract_txt,
}


def reference_docs_node(state: ResearchState) -> dict:
    """First node in the graph. Reads already-extracted zip contents from
    reference_docs_path (the caller saves+unzips before invoking — see
    create_run_workspace), extracts text per file, and writes it to
    <outline_dir>/reference_docs/. No embeddings, no chunking, no vector DB —
    this is a single-shot extraction over a handful of documents, not a
    persistent corpus, so a full-text scan (see match_reference_docs) is
    plenty fast without RAG's overhead."""
    raw_dir = state.get("reference_docs_path")
    if not raw_dir or not os.path.isdir(raw_dir):
        return {"reference_docs": {}}

    extracted_dir = os.path.join(state["outline_dir"], "reference_docs")
    os.makedirs(extracted_dir, exist_ok=True)
    manifest: Dict[str, dict] = {}
    for fname in os.listdir(raw_dir):
        ext = os.path.splitext(fname)[1].lower()
        extractor = REFERENCE_EXTRACTORS.get(ext)
        if not extractor:
            continue
        try:
            text = extractor(os.path.join(raw_dir, fname)).strip()
            text_path = os.path.join(extracted_dir, f"{fname}.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)
            manifest[fname] = {"status": "ok", "text_path": text_path, "char_count": len(text)}
        except Exception as exc:
            manifest[fname] = {"status": "failed", "error": str(exc)}
    
    # Output from this node goes to the plan_node.
    return {"reference_docs": manifest}


def match_reference_docs(item_queries: List[str], sub_topic: str, reference_docs: Dict[str, dict],
                          top_k: int = 2, threshold: float = 0.05) -> List[str]:
    """TF-IDF + cosine similarity: a basic-NLP, embedding-free alternative to
    RAG for matching sections to reference documents. Appropriate at this
    scale (a handful of documents, queried once) — no index, no persistence,
    runs in-memory in milliseconds."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    ok_docs = {f: m for f, m in reference_docs.items() if m.get("status") == "ok"}
    if not ok_docs:
        return []

    doc_names, doc_texts = [], []
    for fname, meta in ok_docs.items():
        with open(meta["text_path"], "r", encoding="utf-8") as f:
            doc_texts.append(f.read())
        doc_names.append(fname)

    query_text = f"{sub_topic} " + " ".join(item_queries or [])
    matrix = TfidfVectorizer(stop_words="english").fit_transform(doc_texts + [query_text])
    sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
    ranked = sorted(zip(doc_names, sims), key=lambda x: -x[1])
    return [name for name, score in ranked[:top_k] if score >= threshold]


def create_run_workspace(reference_zip_path: Optional[str] = None) -> dict:
    """Call this BEFORE app.invoke() — the graph doesn't own run identity or
    zip staging, the caller (Streamlit) does. Returns the fields to merge into
    the initial invoke state: run_id, outline_dir, reference_docs_path."""
    run_id = uuid.uuid4().hex[:8]
    outline_dir = os.path.join(DATA_DIR, run_id)
    raw_dir = os.path.join(outline_dir, "reference_docs_raw")
    os.makedirs(raw_dir, exist_ok=True)
    if reference_zip_path:
        with zipfile.ZipFile(reference_zip_path) as zf:
            zf.extractall(raw_dir)
    return {"run_id": run_id, "outline_dir": outline_dir, "reference_docs_path": raw_dir}
