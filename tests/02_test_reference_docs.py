"""
Test 2: reference_docs_node against a plain folder — no zip needed for
testing. In production, Streamlit calls create_run_workspace() to unzip a
user's upload; here we just point reference_docs_path at a folder we made
ourselves. Drop your own .pdf / .docx / scanned image into sample_data/ to
test those specific extractors too (only .txt and .html are included here
since they need no extra binary test fixtures).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_core import reference_docs_node

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")
OUTLINE_DIR = "/tmp/test_run_refdocs"

if __name__ == "__main__":
    os.makedirs(OUTLINE_DIR, exist_ok=True)
    state = {"reference_docs_path": SAMPLE_DIR, "outline_dir": OUTLINE_DIR}
    result = reference_docs_node(state)

    print("Manifest:")
    for fname, meta in result["reference_docs"].items():
        print(f"  {fname}: {meta}")

    ok = [m for m in result["reference_docs"].values() if m.get("status") == "ok"]
    assert ok, "Expected at least one file to extract successfully"
    for meta in ok:
        assert os.path.exists(meta["text_path"])
        assert meta["char_count"] > 0

    print(f"\n{len(ok)}/{len(result['reference_docs'])} files extracted successfully.")
    print(f"Extracted text saved under: {os.path.join(OUTLINE_DIR, 'reference_docs')}")
