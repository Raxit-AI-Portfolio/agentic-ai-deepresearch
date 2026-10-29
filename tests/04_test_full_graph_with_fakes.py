"""
Test 4: runs the ENTIRE graph — including the interrupt/resume for outline
review — using FakeLLM/FakeSearch/fake_vision_checker. No GPU, no model file,
no API keys. This is the test that proves the graph's WIRING (routing,
validation loop, formatting, export, cleanup) is correct, independent of
model quality. Run this before ever touching the real model or paid APIs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fakes import FakeLLM, FakeSearch, fake_vision_checker
from agentic_core import build_graph, ResearchConfig, ReportConfig, Command, create_run_workspace

if __name__ == "__main__":
    app = build_graph(ResearchConfig(), ReportConfig(),
                       llm=FakeLLM(), search=FakeSearch(), vision_checker=fake_vision_checker)
    thread = {"configurable": {"thread_id": "fake-test-1"}}

    workspace = create_run_workspace()  # no zip — just sets up run_id/outline_dir
    initial_input = {
        **workspace,
        "topic": "Cheese manufacturing in Italy",
        "report_format": "detailed",
        "output_format": "markdown",
        "source_whitelist": [],
        "source_blacklist": [],
    }

    result = app.invoke(initial_input, thread)
    assert "__interrupt__" in result, "Expected the graph to pause for outline review"
    draft_outline = result["__interrupt__"][0].value["draft_outline"]
    print("Paused for review. Draft outline:")
    for item in draft_outline:
        print(" ", item)

    # Simulate the user accepting the outline as-is (no edits, no uploaded content)
    result = app.invoke(Command(resume={"outline": draft_outline, "user_provided_content": {}}), thread)

    assert "final_report" in result, f"Graph did not complete. Keys present: {list(result.keys())}"
    print("\n--- FINAL REPORT ---\n")
    print(result["final_report"])

    print("\nSections generated:", len(result["sections"]))
    for sec in result["sections"]:
        print(f"  - {sec['theme']} / {sec.get('sub_topic')}: status={sec['status']}, media={len(sec.get('media', []))}")

    print("\nAll full-graph (fake) tests passed.")
