"""
Stand-ins for LlamaCpp, GoogleSerperAPIWrapper, and the Gemini vision checker
so the graph's WIRING (routing, validation loop, formatting, export) can be
tested without a GPU, a model file, or any API keys. Swap these for the real
build_llm()/build_search_tool()/make_gemini_vision_checker() only in the final
integration test (05/06 and agentic_core.py's own __main__).

FakeLLM.invoke pattern-matches on the ACTUAL prompt text each node currently
sends (agentic_core/nodes_plan.py, nodes_research.py, content.py) — kept in
sync with those prompts, not with the shape of some earlier version, so
tests 03/04 exercise the real parsing/routing logic instead of silently
falling through to the generic placeholder branch.
"""
import re


class FakeLLM:
    """Mimics LlamaCpp's .invoke(prompt, **kwargs) interface. Pattern-matches
    on prompt content to return a plausible response for whichever node called
    it — good enough to drive every node in the graph at least once."""

    def __init__(self):
        self.call_count = 0
        self.calls = []

    def invoke(self, prompt, **kwargs):
        self.call_count += 1
        self.calls.append(prompt)

        # make_plan_node: one call per theme, expects one pipe-delimited line
        # "Sub-topic Name | query one | query two". Derive a unique sub-topic
        # per theme (from the prompt's own "Theme: ..." line) so plan_node's
        # cross-theme dedup-by-name doesn't collapse every theme into one.
        theme_match = re.search(r"^Theme: (.+)$", prompt, re.MULTILINE)
        if "List 1 to 3 sub-topics" in prompt and theme_match:
            theme_title = theme_match.group(1).strip().title()
            return f"{theme_title} Overview | {theme_title} statistics | {theme_title} trends"

        # make_content_requests_node: one call per candidate sub-topic,
        # expects exactly one line "type | description" (or "none"). Approve
        # a diagram — the only content type that's fully offline (LLM-only,
        # no network/API key) — so non_text_master_node's diagram path is
        # actually exercised by the fakes-only graph test.
        if "A section on this sub-topic could include a data-table" in prompt:
            return "diagram | Sample process diagram"

        if "Does the section contain any claim" in prompt:
            return "OK"

        if "chart spec" in prompt:
            return '{"chart_type": "bar", "title": "Sample Chart", "labels": ["A", "B", "C"], "values": [3, 7, 5]}'

        if "Mermaid" in prompt:
            return "flowchart TD\nA[Start] --> B[End]"

        # make_write_node's pointer-extraction call (standard items only):
        # expects one "<pointer text> | [N]" line per pointer. Return a few
        # deterministic, sub-topic-specific pointers so filter_novel_pointers
        # (services.py) has something real to dedupe across sections even in
        # this fakes-only run — checked before the generic fallback below
        # since that branch would otherwise swallow this prompt too (it also
        # quotes the sub_topic name).
        if "List 4 to 6 short fact pointers" in prompt:
            subject_match = re.search(r'"([^"]+)"', prompt)
            subject = subject_match.group(1) if subject_match else f"topic {self.call_count}"
            return (
                f"{subject} first data point | [1]\n"
                f"{subject} second data point | [2]\n"
                f"{subject} third data point | [none]"
            )

        # write_node's meta/standard prompts both open by quoting the section's
        # theme or sub_topic name in double quotes — use it so different
        # sections get different fake text instead of one literal string for
        # everything. Otherwise every section after the first would score
        # ~100% overlap against prior fake output and trip validate_node's
        # cross-section duplication check (nodes_research.py), forcing every
        # item through the full MAX_REVISIONS loop for no real reason — real
        # models don't produce byte-identical paragraphs across sections.
        # The three sentences deliberately don't share a long literal prefix
        # (a first version used "{subject}: unique details A/B/C.", which
        # drop_self_duplicate_sentences's literal check — ~0.6 threshold —
        # correctly flagged as ~92% self-duplicate, eliminating every
        # section's fake content down to 1 sentence and then out entirely).
        # Real model output doesn't repeat a templated prefix like that
        # across sentences, so this shape is closer to what validate_node
        # actually has to handle.
        subject_match = re.search(r'"([^"]+)"', prompt)
        subject = subject_match.group(1) if subject_match else f"topic {self.call_count}"
        n = self.call_count
        return (
            f"{subject} recorded a {10 + n}% change in reported figures this period. A separate analysis "
            f"placed {subject}'s regional share near {20 + n} percent. Observers tracking {subject} flagged "
            f"{5 + n} notable developments worth following."
        )


class FakeSearch:
    """Mimics GoogleSerperAPIWrapper.results(query)."""

    def results(self, query):
        slug = str(abs(hash(query)))[:8]
        return {"organic": [
            {"title": f"Result for {query}", "link": f"https://example.com/{slug}",
             "snippet": "A sample snippet used only for testing the graph's flow."}
        ]}


def fake_vision_checker(path: str) -> str:
    """Always approves — swap for make_gemini_vision_checker() once you're
    ready to test the real Gemini call."""
    return "OK"
