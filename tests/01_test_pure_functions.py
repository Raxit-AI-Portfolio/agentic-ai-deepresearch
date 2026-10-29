"""
Test 1: pure logic only — no model, no API keys, no files. Run this before
anything else that costs time or money.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_core import ReportConfig, parse_llm_json, _max_overlap_ratio, _linkify_urls, render_chart

if __name__ == "__main__":
    rc = ReportConfig()
    assert rc.get_standard_themes("detailed") == [
        "DETAILED ANALYSIS", "CURRENT TRENDS AND DEVELOPMENTS", "IMPLICATIONS AND RECOMMENDATIONS"
    ]
    print("ReportConfig OK:", rc.themes_for_format("detailed"))

    assert parse_llm_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert parse_llm_json("garbage, not json at all") is None
    print("parse_llm_json OK")

    assert _max_overlap_ratio("the quick brown fox jumps", "the quick brown fox jumps over") > 0.8
    assert _max_overlap_ratio("totally different wording", "unrelated text goes here") < 0.3
    print("_max_overlap_ratio OK")

    linked = _linkify_urls("See https://example.com for more.", {"https://example.com": 1})
    assert linked == "See [1] for more.", linked
    print("_linkify_urls OK:", linked)

    path = render_chart(
        {"chart_type": "bar", "title": "Test Chart", "labels": ["A", "B"], "values": [1, 2]},
        "/tmp/test_chart.png",
    )
    assert os.path.exists(path)
    print("render_chart OK ->", path)

    print("\nAll pure-function tests passed.")
