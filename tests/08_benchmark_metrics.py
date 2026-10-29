"""Test 8: measures the 3 numbers Slide 4 of the portfolio deck was missing
— precision/recall of the deterministic validation checks against a
hand-labeled ground truth set, and (when GOOGLE_API_KEY + SERPER_API_KEY are
set) real end-to-end latency and cost for one minimal live run of the graph.

This is a benchmark, not a pass/fail test — it prints numbers and writes
docs/report.md. Run it directly:

    python tests/08_benchmark_metrics.py

Part A (validation-check accuracy) needs no API key and is always run.
Part B (latency/cost) is skipped, with a clear note, if the keys aren't set.

IMPORTANT about what this does and doesn't measure:
- Part A measures whether this project's OWN deterministic citation/
  fabrication checks (nodes_research.py) correctly flag hand-crafted cases
  with a KNOWN right answer. It is not an evaluation of the LLM's factual
  accuracy against real-world ground truth — no such labeled corpus exists
  in this repo, so that number stays TBD (see docs/report.md).
- Part B's cost is an ESTIMATE from published Gemini 2.5 Flash-Lite pricing
  (https://ai.google.dev/gemini-api/docs/pricing as of 2026-10) multiplied by
  the actual token usage returned by the real API call for one run. It is
  not a billed-invoice number.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must be set before any agentic_core import resolves DATA_DIR (constants.py
# reads DEEP_RESEARCH_DATA_DIR at import time) — keeps run workspaces inside
# the repo instead of under tempfile.gettempdir(), which this sandbox denies
# writes to on a Windows drive root.
_RUNTMP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".runtmp")
os.makedirs(_RUNTMP, exist_ok=True)
os.environ.setdefault("DEEP_RESEARCH_DATA_DIR", os.path.join(_RUNTMP, "deep_research_runs"))

from agentic_core.nodes_research import (  # noqa: E402
    _context_blobs,
    _find_invalid_citation_numbers,
    _find_miscited_sentences,
    _invalid_quarter_claims,
    _MALFORMED_YEAR_CITATION_RE,
    _PLACEHOLDER_RE,
    _spliced_source_titles,
)

REPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "report.md")


# ==========================================
# PART A — validation-check precision/recall against hand-labeled cases
# ==========================================
# Each case is (name, has_issue, check_fn) where check_fn() returns True if
# that check's deterministic function flags the case. "has_issue" is the
# ground-truth label this test file assigns by construction — not derived
# from the code under test.

def _cases():
    cases = []

    # -- empty "( )%"-style placeholder --
    cases.append(("placeholder: empty-paren-percent", True,
                  lambda: bool(_PLACEHOLDER_RE.search("Revenue grew by ( )% last quarter."))))
    cases.append(("placeholder: bare percent-in-parens", True,
                  lambda: bool(_PLACEHOLDER_RE.search("Margin held at (%) for the period."))))
    cases.append(("placeholder: real parenthetical, no issue", False,
                  lambda: bool(_PLACEHOLDER_RE.search("The report (see appendix) covers Q3 in detail."))))
    cases.append(("placeholder: real number in parens, no issue", False,
                  lambda: bool(_PLACEHOLDER_RE.search("The share grew by (12%) last quarter."))))

    # -- invalid "Nth quarter" (quarters only run 1-4) --
    cases.append(("quarter: 5th quarter is invalid", True,
                  lambda: bool(_invalid_quarter_claims("Growth was strongest in the 5th quarter of fiscal 2024."))))
    cases.append(("quarter: 23rd quarter is invalid", True,
                  lambda: bool(_invalid_quarter_claims("Sales peaked in the 23rd quarter."))))
    cases.append(("quarter: 3rd quarter is valid, no issue", False,
                  lambda: bool(_invalid_quarter_claims("Growth was strongest in the 3rd quarter of fiscal 2024."))))
    cases.append(("quarter: 1st quarter is valid, no issue", False,
                  lambda: bool(_invalid_quarter_claims("Revenue dipped in the 1st quarter."))))

    # -- "the year [N]" where a year was meant, a citation bracket landed instead --
    cases.append(("malformed year+citation: 'the year [N]'", True,
                  lambda: bool(_MALFORMED_YEAR_CITATION_RE.search("This happened in the year [12]."))))
    cases.append(("malformed year+citation: real year + citation, no issue", False,
                  lambda: bool(_MALFORMED_YEAR_CITATION_RE.search("This happened in 2012 [12]."))))

    # -- citation number not present in the item's own Research Notes --
    cases.append(("invalid citation number: [5] not in {1,2,3}", True,
                  lambda: bool(_find_invalid_citation_numbers("Revenue rose sharply[5].", {1, 2, 3}))))
    cases.append(("invalid citation number: [2] is in {1,2,3}, no issue", False,
                  lambda: bool(_find_invalid_citation_numbers("Revenue rose sharply[2].", {1, 2, 3}))))

    # -- source title copied near-verbatim into drafted prose --
    long_title = "Global Smartphone Shipments Decline For Third Consecutive Quarter In 2024"
    cases.append(("title splice: verbatim title copied into prose", True,
                  lambda: bool(_spliced_source_titles(
                      f"According to one report, {long_title.lower()} across every major region.",
                      [{"title": long_title, "url": "https://example.com/a"}]))))
    cases.append(("title splice: paraphrased, no issue", False,
                  lambda: bool(_spliced_source_titles(
                      "Smartphone shipments fell again in 2024, extending a multi-quarter slump.",
                      [{"title": long_title, "url": "https://example.com/a"}]))))

    # -- sentence reads closer (TF-IDF/cosine) to a different source than the one it cites --
    context = (
        "[1] Title: Electric vehicle battery costs fall sharply\n"
        "[1] URL: https://example.com/1\n"
        "[1] Summary: Lithium-ion battery pack prices dropped due to cheaper raw materials and scale.\n"
        "[2] Title: Airline ticket prices rise on fuel costs\n"
        "[2] URL: https://example.com/2\n"
        "[2] Summary: Jet fuel price increases pushed airlines to raise ticket prices across most routes.\n"
    )
    blobs = _context_blobs(context)
    cases.append(("miscitation: cites [2] but content matches [1]'s topic", True,
                  lambda: bool(_find_miscited_sentences(
                      "Battery pack prices fell due to cheaper lithium-ion raw materials and scale[2].", blobs))))
    cases.append(("miscitation: cites [1] and matches [1]'s topic, no issue", False,
                  lambda: bool(_find_miscited_sentences(
                      "Battery pack prices fell due to cheaper lithium-ion raw materials and scale[1].", blobs))))

    return cases


def run_part_a():
    cases = _cases()
    tp = fp = tn = fn = 0
    rows = []
    for name, expected, check_fn in cases:
        actual = check_fn()
        outcome = "TP" if (expected and actual) else "TN" if (not expected and not actual) \
            else "FP" if (not expected and actual) else "FN"
        if outcome == "TP":
            tp += 1
        elif outcome == "TN":
            tn += 1
        elif outcome == "FP":
            fp += 1
        else:
            fn += 1
        rows.append((name, expected, actual, outcome))

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else float("nan")
    accuracy = (tp + tn) / len(cases)

    print("\n=== Part A: validation-check precision/recall (hand-labeled ground truth) ===")
    for name, expected, actual, outcome in rows:
        print(f"  [{outcome}] {name}: expected={expected} actual={actual}")
    print(f"\n  n={len(cases)}  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  accuracy={accuracy:.2f}  precision={precision:.2f}  recall={recall:.2f}  f1={f1:.2f}")

    return {"cases": rows, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# ==========================================
# PART B — real end-to-end latency + cost estimate (needs real API keys)
# ==========================================

class MeasuredLLM:
    """Wraps the real GeminiLLM (from build_llm) to time each call and
    capture token usage for a cost estimate — bypasses GeminiLLM's own
    retry wrapper so usage_metadata on the raw response is reachable; this
    is a benchmark harness, not production code, so no retry-on-failure
    here is an accepted tradeoff."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = []

    def invoke(self, prompt, max_tokens=None, **kwargs):
        if max_tokens is not None:
            kwargs.setdefault("max_output_tokens", max_tokens)
        t0 = time.perf_counter()
        resp = self._inner._chat.invoke(prompt, **kwargs)
        dt = time.perf_counter() - t0
        usage = getattr(resp, "usage_metadata", None) or {}
        self.calls.append({
            "latency_s": dt,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        })
        return resp.content


class MeasuredSearch:
    """Wraps the real GoogleSerperAPIWrapper to time each .results() call."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = []

    def results(self, query):
        t0 = time.perf_counter()
        r = self._inner.results(query)
        dt = time.perf_counter() - t0
        self.calls.append({"query": query, "latency_s": dt})
        return r


# Published Gemini 2.5 Flash-Lite pay-as-you-go pricing, source:
# https://ai.google.dev/gemini-api/docs/pricing (checked 2026-10-28). An
# estimate multiplied by real measured token counts, not a billed invoice.
GEMINI_FLASH_LITE_PRICE_PER_M_INPUT_TOKENS = 0.10
GEMINI_FLASH_LITE_PRICE_PER_M_OUTPUT_TOKENS = 0.40


def run_part_b():
    have_google = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    have_serper = bool(os.getenv("SERPER_API_KEY"))
    if not (have_google and have_serper):
        print("\n=== Part B: skipped — GOOGLE_API_KEY and/or SERPER_API_KEY not set ===")
        return None

    from agentic_core import (
        Command,
        ReportConfig,
        ResearchConfig,
        build_graph,
        build_llm,
        build_search_tool,
        create_run_workspace,
        noop_vision_checker,
    )

    config = ResearchConfig(llm_provider="gemini-2.5-flash-lite")
    measured_llm = MeasuredLLM(build_llm(config))
    measured_search = MeasuredSearch(build_search_tool(config))

    # report_format left as an unrecognized string -> config.py's
    # themes_for_format() fallback list (INTRODUCTION/DETAILED ANALYSIS/
    # CONCLUSION: 2 meta + 1 standard) — smallest work queue that still
    # exercises the research+write+validate loop once, to keep real API
    # spend minimal while still measuring every node type.
    app = build_graph(config, ReportConfig(), llm=measured_llm, search=measured_search,
                       vision_checker=noop_vision_checker)
    thread = {"configurable": {"thread_id": "benchmark-1"}}
    workspace = create_run_workspace(reference_zip_path=None)
    initial_input = {
        **workspace,
        "topic": "Benchmark run: renewable energy adoption",
        "report_format": "benchmark-minimal",
        "output_format": "markdown",
        "source_whitelist": [],
        "source_blacklist": [],
    }

    print("\n=== Part B: real end-to-end run (gemini-2.5-flash-lite + Serper) ===")
    t0 = time.perf_counter()
    result = app.invoke(initial_input, thread)
    if "__interrupt__" in result:
        draft_outline = result["__interrupt__"][0].value["draft_outline"]
        result = app.invoke(Command(resume={"outline": draft_outline, "user_provided_content": {}}), thread)
    total_latency = time.perf_counter() - t0

    input_tokens = sum(c["input_tokens"] or 0 for c in measured_llm.calls)
    output_tokens = sum(c["output_tokens"] or 0 for c in measured_llm.calls)
    est_cost = (input_tokens / 1_000_000 * GEMINI_FLASH_LITE_PRICE_PER_M_INPUT_TOKENS
                + output_tokens / 1_000_000 * GEMINI_FLASH_LITE_PRICE_PER_M_OUTPUT_TOKENS)

    print(f"  total latency: {total_latency:.1f}s")
    print(f"  LLM calls: {len(measured_llm.calls)}  search calls: {len(measured_search.calls)}")
    print(f"  tokens: input={input_tokens} output={output_tokens}")
    print(f"  estimated cost: ${est_cost:.5f}")
    print(f"  sections generated: {len(result.get('sections', []))}")

    return {
        "total_latency_s": total_latency,
        "llm_calls": measured_llm.calls,
        "search_calls": measured_search.calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": est_cost,
        "sections_generated": len(result.get("sections", [])),
    }


def write_report(part_a, part_b):
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    lines = []
    lines.append("# Benchmark report")
    lines.append("")
    lines.append(f"Generated by `tests/08_benchmark_metrics.py` on {time.strftime('%Y-%m-%d')}.")
    lines.append("")
    lines.append("## Part A — validation-check accuracy against hand-labeled ground truth")
    lines.append("")
    lines.append(
        "Measures whether this project's own deterministic checks in `nodes_research.py` "
        "(`_find_invalid_citation_numbers`, `_find_miscited_sentences`, `_spliced_source_titles`, "
        "`_invalid_quarter_claims`, the placeholder and malformed-year-citation regexes) correctly "
        "flag hand-crafted cases with a known right answer. This is NOT an evaluation of the LLM's "
        "factual accuracy against real-world ground truth — no such labeled corpus exists in this "
        "repo yet (see `open-issues/future-extensions.md`)."
    )
    lines.append("")
    lines.append(f"- Cases: {len(part_a['cases'])}")
    lines.append(f"- TP={part_a['tp']} FP={part_a['fp']} TN={part_a['tn']} FN={part_a['fn']}")
    lines.append(f"- Accuracy: {part_a['accuracy']:.2f}")
    lines.append(f"- Precision: {part_a['precision']:.2f}")
    lines.append(f"- Recall: {part_a['recall']:.2f}")
    lines.append(f"- F1: {part_a['f1']:.2f}")
    lines.append("")
    lines.append("| Case | Expected | Actual | Outcome |")
    lines.append("|---|---|---|---|")
    for name, expected, actual, outcome in part_a["cases"]:
        lines.append(f"| {name} | {expected} | {actual} | {outcome} |")
    lines.append("")
    lines.append("## Part B — real end-to-end latency and estimated cost")
    lines.append("")
    if part_b is None:
        lines.append(
            "Skipped — `GOOGLE_API_KEY` and/or `SERPER_API_KEY` not set when this report was "
            "generated. Latency and cost are TBD until re-run with both keys present."
        )
    else:
        lines.append(
            "One real run through the full graph (`gemini-2.5-flash-lite` + Serper, no fakes), "
            "topic \"Benchmark run: renewable energy adoption\", using the 3-theme fallback report "
            f"format (an unrecognized `report_format` so `themes_for_format()` falls back to "
            f"INTRODUCTION/DETAILED ANALYSIS/CONCLUSION — plan_node expanded DETAILED ANALYSIS into "
            f"multiple sub-topics, so this run produced {part_b['sections_generated']} sections "
            f"total, not a fixed count chosen in advance)."
        )
        lines.append("")
        lines.append(f"- Total wall-clock latency: {part_b['total_latency_s']:.1f}s")
        lines.append(f"- Sections generated: {part_b['sections_generated']}")
        lines.append(f"- LLM calls: {len(part_b['llm_calls'])}, search calls: {len(part_b['search_calls'])}")
        lines.append(f"- Tokens: input={part_b['input_tokens']}, output={part_b['output_tokens']}")
        lines.append(
            f"- Estimated cost: ${part_b['estimated_cost_usd']:.5f} "
            f"(published Gemini 2.5 Flash-Lite pricing — $0.10/M input, $0.40/M output tokens, "
            f"https://ai.google.dev/gemini-api/docs/pricing, checked 2026-10-28 — "
            f"multiplied by the actual measured token counts above; not a billed invoice)"
        )
        lines.append("")
        lines.append("Per-call detail:")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps({"llm_calls": part_b["llm_calls"], "search_calls": part_b["search_calls"]}, indent=2))
        lines.append("```")
    lines.append("")
    lines.append("## What's still not measured")
    lines.append("")
    lines.append(
        "- LLM factual accuracy against a real-world ground-truth corpus (no such corpus exists in "
        "this repo) — stays TBD.\n"
        "- Cost/latency at larger report scale (more topics, more sections per run, or the "
        "'detailed'/'academic'/'presentation' formats specifically) — this run used an "
        "unrecognized `report_format` to land on the smallest theme set and limit real API spend; "
        "re-run with a real format name for a larger-scale number.\n"
        "- Cost under the local `gemma_7b` provider (no API cost, but needs the GPU/CPU time instead "
        "— not measured here)."
    )
    lines.append("")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    part_a = run_part_a()
    part_b = run_part_b()
    write_report(part_a, part_b)
    print("\nBenchmark complete.")
