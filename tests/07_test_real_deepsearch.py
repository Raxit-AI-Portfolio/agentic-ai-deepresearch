import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_core import (
    build_graph, ResearchConfig, ReportConfig, Command,
    create_run_workspace, build_llm, build_search_tool, add_content_request,
)
from fakes import fake_vision_checker

# --- Real LLM + real search, fake vision checker ---
config = ResearchConfig(
    model_path="/content/drive/MyDrive/AI/models/gemma-7b-it-Q4_K_M.gguf",   # must exist and be >1GB (validate_model_file)
    # search_api_key defaults to os.getenv("SERPER_API_KEY", "") — make sure that's set
)
llm = build_llm(config)              # raises FileNotFoundError if model_path is missing
search = build_search_tool(config)   # GoogleSerperAPIWrapper — needs SERPER_API_KEY

app = build_graph(config, ReportConfig(), llm=llm, search=search, vision_checker=fake_vision_checker)
thread = {"configurable": {"thread_id": "real-llm-test-1"}}

# --- Step 1: kick off the run ---
workspace = create_run_workspace(reference_zip_path=None)
initial_input = {
    **workspace,
    "topic": "Car type wise sales trends in the Germany in the year 2025",
    "report_format": "presentation",
    "output_format": "markdown",
    "source_whitelist": [],
    "source_blacklist": ["pinterest.com","facebook.com", "instagram.com","youtube.com","twitter.com","reddit.com"],
}
result = app.invoke(initial_input, thread)   # pauses at review_outline

# --- Step 2: inspect / edit the outline (still same process) ---
draft_outline = result["__interrupt__"][0].value["draft_outline"]
print(json.dumps(draft_outline, indent=2))
# edit draft_outline in-place here if you want to change sub_topics/queries/weightage

# Manually force content requests, to test each generator/render path
# directly, independent of whether content_requests_node's LLM call ever
# approves one on its own. add_content_request() is the standard way to add
# a content request by hand — it finds the outline item by sub_topic name and
# builds the {"type", "description", ...} dict for you, raising a clear error
# on a typo'd type or sub_topic instead of silently doing nothing.
contents_dir = os.path.join(workspace["outline_dir"], "contents")
os.makedirs(contents_dir, exist_ok=True)

# 1) internet-image — no user_supplied_path, so this exercises the real
#    Unsplash fetch (needs UNSPLASH_ACCESS_KEY set).
# add_content_request(draft_outline, draft_outline[0]["sub_topic"],
#                      "internet-image", "Boeing 737 MAX aircraft on the tarmac, with a dramatic sky in the background")

# 2) data-table — user-supplied, tests the markdown-table render path
#    independent of the Wikipedia-search fetch (which needs a real matching
#    Wikipedia page with an actual <table> to succeed).
# table_path = os.path.join(contents_dir, "cost_savings.md")
# with open(table_path, "w", encoding="utf-8") as f:
#     f.write(
#         "| Strategy | Typical Savings |\n"
#         "|---|---|\n"
#         "| Process automation | 20-37% |\n"
#         "| Remote/hybrid work | 10-15% |\n"
#         "| Energy-efficient equipment | 5-20% |\n"
#     )
# add_content_request(draft_outline, draft_outline[1]["sub_topic"],
#                      "data-table", "Cost-saving strategy comparison", user_supplied_path=table_path)

# 3) diagram — LLM-GENERATED: only a description, no user_supplied_path. This
# is the real make_diagram_content(llm) path — it prompts Gemma for raw
# Mermaid syntax from the description below and validates the response
# against that type's own strict grammar (see DIAGRAM_CONFIGS in content.py).
# diagram_type defaults to "flowchart" when omitted — passed explicitly here
# just to show the parameter exists. If the model's output doesn't pass
# validation, format_sections_node renders a TODO comment instead of
# embedding broken Mermaid — so either way this run tells you concretely
# whether Gemma can write valid Mermaid for this description, without guessing.
# add_content_request(draft_outline, draft_outline[2]["sub_topic"],
#                      "diagram", "flowchart as timeline of mobile phone sensor evolution, mentioning years and key sensor types",
#                      diagram_type="flowchart")

# 3b) diagram — same LLM-generated path, but a different Mermaid type
# (mindmap). Available types: flowchart, sequenceDiagram, classDiagram,
# mindmap — see DIAGRAM_CONFIGS. (Guarded: a different topic/run may produce
# fewer than 4 sub-topics.)
# if len(draft_outline) > 3:
#     add_content_request(draft_outline, draft_outline[3]["sub_topic"],
#                          "diagram", "mind map of mobile phone sensors as root node, with branches for each sensor type and sub-branches for sub types",
#                          diagram_type="mindmap")


# --- Step 3 (optional): stage a content zip before resuming ---
# SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")
# SAMPLE_ZIP = os.path.join(SAMPLE_DIR, "user_content.zip")
# with zipfile.ZipFile(SAMPLE_ZIP) as zf:
#     zf.extractall(os.path.join(workspace["outline_dir"], "contents"))
user_provided_content = {}   # or {"cheese_map.png": {"theme": "...", "sub_topic": "...", "type": "internet-image"}}

# --- Step 4: resume, same thread, same process ---
result = app.invoke(Command(resume={"outline": draft_outline, "user_provided_content": user_provided_content}), thread)

print(result["final_report"])
if result.get("export_path"):
    print("Exported to:", result["export_path"])