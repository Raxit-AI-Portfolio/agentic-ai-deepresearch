"""
Test 3: drives individual graph nodes directly with a hand-built state dict —
no LangGraph wiring involved yet. Good for checking one node's logic in
isolation before trusting the full graph in test 4.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fakes import FakeLLM, FakeSearch
from agentic_core import make_plan_node, ReportConfig, build_work_queue_node, make_research_node

if __name__ == "__main__":
    llm = FakeLLM()
    report_cfg = ReportConfig()

    # --- plan_node ---
    plan_node = make_plan_node(llm, report_cfg)
    state = {"topic": "Cheese in Italy", "report_format": "detailed",
             "outline_dir": "/tmp/test_run_nodes", "run_id": "test1", "reference_docs": {}}
    os.makedirs(state["outline_dir"], exist_ok=True)
    plan_result = plan_node(state)
    print("draft_outline:")
    for item in plan_result["draft_outline"]:
        print(" ", item)
    state.update(plan_result)

    # --- build_work_queue_node (skip review_outline's interrupt for this isolated test —
    # note weightage will be 0.0 for every item since only review_outline_node fills
    # in the default split; that's expected here, not a bug) ---
    state["reviewed_outline"] = state["draft_outline"]
    state["user_provided_content"] = {}
    wq_result = build_work_queue_node(state)
    print("\nwork_queue:")
    for item in wq_result["work_queue"]:
        print(" ", item)
    state.update(wq_result)

    # --- research_node ---
    # Mirrors real routing: research only ever runs for a 'standard' item at
    # the front of the queue (route_next sends 'meta' items straight to write).
    standard_queue = [i for i in wq_result["work_queue"] if i["kind"] == "standard"]
    state["work_queue"] = standard_queue
    research_node = make_research_node(FakeSearch())
    state["source_whitelist"], state["source_blacklist"] = [], []
    research_result = research_node(state)
    print("\ncurrent_context:\n", research_result["current_context"])
    print("\ncurrent_sources:", research_result["current_sources"])

    assert len(plan_result["draft_outline"]) >= 1
    assert len(wq_result["work_queue"]) >= 1
    assert research_result["current_context"]
    print("\nAll single-node tests passed.")
