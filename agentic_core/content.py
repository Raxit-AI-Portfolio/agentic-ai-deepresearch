"""Non-text content generators (the "child nodes").

Chart content is NOT LLM-generated code. Gemma only produces a small JSON
chart spec (type/labels/values); a fixed, deterministic matplotlib function
renders it. No generated code is ever executed — this avoids both the
reliability problem (small models are weak at code-gen) and the security
problem (executing LLM output is a real attack surface regardless of model).

internet-image needs UNSPLASH_ACCESS_KEY (licensed source, not a raw image
search, to keep licensing unambiguous). ai-generated-image is a stub — wire
it to Stable Diffusion/FLUX or a hosted image-gen API. data-table fetch is
deterministic (pandas.read_html), no LLM involved, so no hallucinated
numbers."""

from __future__ import annotations

import json
import os
import re
import tempfile
from typing import List, Literal, Optional, TypedDict

import requests
from langchain_community.llms import LlamaCpp
from langchain_community.utilities import GoogleSerperAPIWrapper

from .services import parse_llm_json

# Applies to every generated image (Unsplash photos, rendered charts) — keeps
# the report from embedding multi-megabyte source photos regardless of what
# size the source happens to serve.
MAX_IMAGE_DIMENSION = 800  # pixels, longest side
IMAGE_JPEG_QUALITY = 85


def _resize_image_file(path: str, max_dimension: int = MAX_IMAGE_DIMENSION,
                        quality: int = IMAGE_JPEG_QUALITY) -> None:
    """Downscales an image file in place so its longest side is at most
    max_dimension, and re-saves as a compressed JPEG. thumbnail() is a no-op
    if the image is already smaller, so this is safe to call unconditionally."""
    from PIL import Image as PILImage
    with PILImage.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_dimension, max_dimension))
        img.save(path, "JPEG", quality=quality, optimize=True)


class ContentRequest(TypedDict, total=False):
    # ai-generated-image (stub, unimplemented) and chart (never receives real
    # data — see make_chart_content) are excluded from what plan_node's
    # prompt offers; their generator functions stay below in case either is
    # wired up properly later.
    type: Literal["internet-image", "data-table", "diagram"]
    description: str
    user_supplied_path: Optional[str]
    # Only meaningful when type == "diagram" — one of DIAGRAM_CONFIGS' keys
    # (flowchart/sequenceDiagram/classDiagram/mindmap). Defaults to
    # "flowchart" when absent, since content_requests_node's automatic LLM
    # decision only ever says "diagram" generically; this is for manual use
    # via add_content_request(..., diagram_type=...).
    diagram_type: Optional[str]


class ContentResult(TypedDict, total=False):
    type: str
    path: Optional[str]          # local file: images / rendered charts
    markup: Optional[str]        # embeddable text/code: mermaid diagrams, markdown tables
    caption: str
    source: Literal["user_supplied", "generated"]
    status: Literal["pending", "approved", "rejected"]
    validation_notes: Optional[str]


def internet_image_content(description: str) -> ContentResult:
    """Sources from Unsplash (licensed, not a raw image search) so licensing is
    unambiguous at fetch time rather than inferred later."""
    api_key = os.getenv("UNSPLASH_ACCESS_KEY")
    if not api_key:
        return {"type": "internet-image", "path": None, "caption": description,
                "source": "generated", "status": "pending",
                "validation_notes": "UNSPLASH_ACCESS_KEY not set — image not fetched."}
    try:
        resp = requests.get("https://api.unsplash.com/search/photos",
                             params={"query": description, "per_page": 1},
                             headers={"Authorization": f"Client-ID {api_key}"}, timeout=10)
        data = resp.json()
        if resp.status_code != 200:
            errors = data.get("errors") if isinstance(data, dict) else None
            note = f"Unsplash API error ({resp.status_code}): {'; '.join(errors) if errors else data}"
            return {"type": "internet-image", "path": None, "caption": description,
                     "source": "generated", "status": "pending", "validation_notes": note}
        results = data.get("results", [])
        if not results:
            return {"type": "internet-image", "path": None, "caption": description,
                     "source": "generated", "status": "pending", "validation_notes": "No results found."}
        urls = results[0]["urls"]
        url = urls.get("small") or urls["regular"]  # smaller download; _resize_image_file caps it further regardless
        path = os.path.join(tempfile.gettempdir(), f"img_{abs(hash(description))}.jpg")
        with open(path, "wb") as f:
            f.write(requests.get(url, timeout=10).content)
        _resize_image_file(path)
        return {"type": "internet-image", "path": path, "caption": description,
                 "source": "generated", "status": "pending"}
    except Exception as exc:
        return {"type": "internet-image", "path": None, "caption": description,
                 "source": "generated", "status": "pending", "validation_notes": f"Fetch failed: {exc}"}


def ai_generated_image_content(description: str) -> ContentResult:
    """Stub — wire to Stable Diffusion/FLUX or a hosted image-gen API
    (e.g. Google's Nano Banana) here."""
    return {"type": "ai-generated-image", "path": None, "caption": description,
            "source": "generated", "status": "pending",
            "validation_notes": "Image-gen backend not wired yet."}


def make_data_table_content(search: GoogleSerperAPIWrapper):
    """Deterministic: finds a source page, parses its largest HTML table with
    pandas. No LLM involved, so no hallucinated numbers."""

    def data_table_content(description: str) -> ContentResult:
        try:
            import io
            import pandas as pd
            data = search.results(f"{description} site:wikipedia.org")
            organic = data.get("organic", [])
            if not organic:
                raise ValueError("No source page found.")
            url = organic[0]["link"]
            tables = pd.read_html(io.StringIO(requests.get(url, timeout=10).text))
            if not tables:
                raise ValueError("No tables found on source page.")
            table = max(tables, key=len).head(15)
            return {"type": "data-table", "path": None, "markup": table.to_markdown(index=False),
                     "caption": f"{description} (source: {url})", "source": "generated", "status": "pending"}
        except Exception as exc:
            return {"type": "data-table", "path": None, "caption": description,
                     "source": "generated", "status": "pending",
                     "validation_notes": f"Fetch/parse failed: {exc} (needs pandas + tabulate)"}

    return data_table_content


def render_chart(spec: dict, out_path: str) -> str:
    """Deterministic rendering of a fixed set of chart types from an LLM-chosen
    JSON spec. No generated code is ever executed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    kind = spec.get("chart_type", "bar")
    labels, values = spec.get("labels", []), spec.get("values", [])
    if kind == "pie":
        ax.pie(values, labels=labels, autopct="%1.1f%%")
    elif kind == "line":
        ax.plot(labels, values, marker="o")
        ax.set_xlabel(spec.get("x_label", "")); ax.set_ylabel(spec.get("y_label", ""))
    elif kind == "scatter":
        ax.scatter(labels, values)
        ax.set_xlabel(spec.get("x_label", "")); ax.set_ylabel(spec.get("y_label", ""))
    else:
        ax.bar(labels, values)
        ax.set_xlabel(spec.get("x_label", "")); ax.set_ylabel(spec.get("y_label", ""))
    ax.set_title(spec.get("title", ""))
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)  # 6in figure * 130dpi = 780px, under MAX_IMAGE_DIMENSION
    plt.close(fig)
    return out_path


def make_chart_content(llm: LlamaCpp):
    def chart_content(description: str, table_data: Optional[dict] = None) -> ContentResult:
        if table_data:
            spec_prompt = f"""<start_of_turn>user
Using this data: {json.dumps(table_data)}
Produce a JSON chart spec for: "{description}"
Return ONLY JSON: {{"chart_type": "pie|bar|line|scatter", "title": "...", "labels": [...], "values": [...], "x_label": "...", "y_label": "..."}}
<end_of_turn>
<start_of_turn>model
"""
        else:
            # No real dataset available — do NOT invent numbers. Skip the chart
            # rather than fabricate data presented as fact.
            return {"type": "chart", "path": None, "caption": description,
                     "source": "generated", "status": "pending",
                     "validation_notes": "No underlying data available — skipped rather than fabricate values."}

        response = llm.invoke(spec_prompt, max_tokens=300)
        spec = parse_llm_json(response)
        if not spec or "values" not in spec:
            return {"type": "chart", "path": None, "caption": description,
                     "source": "generated", "status": "pending", "validation_notes": "Could not derive a chart spec."}
        path = render_chart(spec, os.path.join(tempfile.gettempdir(), f"chart_{abs(hash(description))}.png"))
        return {"type": "chart", "path": path, "caption": spec.get("title", description),
                 "source": "generated", "status": "pending"}

    return chart_content


# --------------------------------------------------------------------------
# Mermaid diagram generation — multiple diagram types, each with its own
# prompt example and its own strict validator. Checking only the opening
# keyword (or even that an arrow token appears ANYWHERE in the text) isn't
# enough — observed directly against real output twice: once with a numbered
# prose list following "flowchart" with no arrows at all, and once with ONE
# genuinely valid edge line followed by garbage that a single-arrow-anywhere
# check happily let through. Each type below validates the FULL structure —
# a real declaration line, followed by ONLY well-formed lines for that
# type's grammar, checked individually, so one malformed line rejects the
# whole diagram rather than being missed.
# --------------------------------------------------------------------------

_MERMAID_NAME = r"[A-Za-z0-9_]+"
_MERMAID_FLOW_NODE = rf"{_MERMAID_NAME}(\[[^\[\]]*\]|\([^()]*\)|\{{[^{{}}]*\}})?"

# A prose list dressed up with markdown bold / HTML tags is invalid in every
# diagram type — checked regardless of which type is chosen.
_MERMAID_JUNK_RE = re.compile(r"\*\*|<br\s*/?>", re.IGNORECASE)
_MERMAID_NUMBERED_LIST_RE = re.compile(r"^\d+[\.\)]\s")


def _validate_mindmap(body_lines: List[str]) -> tuple[bool, Optional[str]]:
    """Mindmap has no arrows at all — hierarchy is expressed purely through
    indentation — so it needs a different check than the arrow-based types."""
    if len(body_lines) < 2:
        return False, "No content after the mindmap declaration line."
    indents = set()
    for line in body_lines:
        if _MERMAID_NUMBERED_LIST_RE.match(line.strip()):
            return False, f"Line looks like a numbered prose list, not a node label: {line!r}"
        stripped = line.strip()
        # A bare colon (not part of a ((shape)) marker) reads as a "Label:
        # explanation" prose line, not a real mindmap node.
        if ":" in stripped and not re.search(r"\(\(|\)\)", stripped):
            return False, f"Line looks like prose (contains ':'), not a short node label: {line!r}"
        if len(stripped) > 80:
            return False, f"Line is too long to be a node label: {line!r}"
        indents.add(len(line) - len(line.lstrip(" ")))
    if len(indents) < 2:
        return False, "All lines have the same indentation — no hierarchy, not a real mind map."
    return True, None


DIAGRAM_CONFIGS = {
    "flowchart": {
        "declaration_re": re.compile(r"^(flowchart|graph)\s+(TD|TB|BT|RL|LR)\b", re.IGNORECASE),
        "line_res": [
            re.compile(rf"^{_MERMAID_FLOW_NODE}\s*(-->|---|-\.->|==>)\s*(\|[^|]*\|\s*)?{_MERMAID_FLOW_NODE}$"),
        ],
        "example": (
            "flowchart TD\n"
            "    A[Collect Requirements] --> B[Design Solution]\n"
            "    B --> C[Implement]\n"
            "    C --> D[Test]\n"
            "    D --> E[Deploy]"
        ),
        "instruction": ('Each line must be either the "flowchart TD" declaration or a node '
                         'connected to another node with an arrow (-->).'),
    },
    "sequenceDiagram": {
        "declaration_re": re.compile(r"^sequenceDiagram\b", re.IGNORECASE),
        "line_res": [
            re.compile(rf"^{_MERMAID_NAME}(\s+{_MERMAID_NAME})*\s*(->>|-->>|--x|-x|->|-->)\s*"
                       rf"{_MERMAID_NAME}(\s+{_MERMAID_NAME})*\s*:\s*.+$"),
            re.compile(r"^(participant|actor|note|activate|deactivate|loop|end|alt|else|"
                       r"opt|par|and|critical|rect|autonumber)\b", re.IGNORECASE),
        ],
        "example": (
            "sequenceDiagram\n"
            "    Customer->>Support: Submit ticket\n"
            "    Support->>Engineer: Escalate issue\n"
            "    Engineer-->>Support: Provide fix\n"
            "    Support-->>Customer: Resolve ticket"
        ),
        "instruction": ('Each line must be either the "sequenceDiagram" declaration or a message '
                         'between two participants using an arrow (->> or -->>), '
                         'like "Alice->>Bob: message text".'),
    },
    "classDiagram": {
        "declaration_re": re.compile(r"^classDiagram\b", re.IGNORECASE),
        "line_res": [
            re.compile(rf'^"?{_MERMAID_NAME}"?\s*(<\|--|\*--|o--|-->|\.\.>|\.\.\|>)\s*"?{_MERMAID_NAME}"?(\s*:\s*.+)?$'),
            re.compile(rf"^{_MERMAID_NAME}\s*:\s*.+$"),
            re.compile(r"^(class|<<interface>>|<<abstract>>)\b", re.IGNORECASE),
        ],
        "example": (
            "classDiagram\n"
            "    Animal <|-- Duck\n"
            "    Animal <|-- Fish\n"
            "    Animal : +String name\n"
            "    Animal : +move()"
        ),
        "instruction": ('Each line must be either the "classDiagram" declaration, a relationship '
                         'between two classes (<|--, *--, o--, -->), or a member line '
                         'like "ClassName : +methodName()".'),
    },
    "mindmap": {
        "declaration_re": re.compile(r"^mindmap\b", re.IGNORECASE),
        "custom_validator": _validate_mindmap,
        "example": (
            "mindmap\n"
            "  root((Cost Saving Strategies))\n"
            "    Operational Efficiency\n"
            "      Process Automation\n"
            "      Inventory Control\n"
            "    Marketing\n"
            "      Digital Advertising\n"
            "      Social Media\n"
            "    Technology\n"
            "      Cloud Computing\n"
            "      AI Tools"
        ),
        "instruction": ('Each line must be either the "mindmap" declaration or a short node label. '
                         'Use INDENTATION (extra leading spaces) to show a node is a child of the '
                         'less-indented node above it — do not use arrows, colons, or full sentences.'),
    },
}


_RISKY_SQUARE_LABEL_RE = re.compile(r'\[([^\[\]"]*[()][^\[\]"]*)\]')
_RISKY_DIAMOND_LABEL_RE = re.compile(r'\{([^{}"]*[()][^{}"]*)\}')


def _quote_risky_flowchart_labels(code: str) -> str:
    """Mermaid's real flowchart grammar treats a literal '(' or ')' inside an
    unquoted square-bracket OR curly-brace node label as the start of a
    DIFFERENT node shape and fails to parse — confirmed directly by running
    generated diagrams through mermaid's own parser: a label like
    "[Future: Explainable AI (XAI) in Trading]" produced a hard parse error
    ("Expecting ... got 'PS'"), and the same failure recurred later for a
    diamond/decision node, "{AI for Visual Effects (VFX)}" — even though both
    passed this module's own validate_diagram (whose _MERMAID_FLOW_NODE regex
    is deliberately permissive about label content and doesn't know about
    this real-Mermaid restriction). Wrapping such labels in double quotes is
    Mermaid's own documented escape hatch and fixes the diagram instead of
    losing it."""
    code = _RISKY_SQUARE_LABEL_RE.sub(lambda m: f'["{m.group(1)}"]', code)
    code = _RISKY_DIAMOND_LABEL_RE.sub(lambda m: f'{{"{m.group(1)}"}}', code)
    return code


def validate_diagram(code: str, diagram_type: str) -> tuple[bool, Optional[str]]:
    cfg = DIAGRAM_CONFIGS[diagram_type]
    if _MERMAID_JUNK_RE.search(code):
        return False, "Contains markdown bold or HTML tags — not real Mermaid syntax."
    raw_lines = code.splitlines()
    non_empty = [l for l in raw_lines if l.strip()]
    if not non_empty or not cfg["declaration_re"].match(non_empty[0].strip()):
        return False, f'First line is not a valid "{diagram_type}" declaration.'

    if "custom_validator" in cfg:
        # Pass raw (un-stripped) lines after the declaration — mindmap's
        # validator needs the original leading whitespace to check indentation.
        decl_index = raw_lines.index(non_empty[0])
        body = [l for l in raw_lines[decl_index + 1:] if l.strip()]
        return cfg["custom_validator"](body)

    body = non_empty[1:]
    if not body:
        return False, "No content after the declaration line."
    for line in body:
        if not any(r.match(line.strip()) for r in cfg["line_res"]):
            return False, f"Line does not match any valid pattern for {diagram_type}: {line!r}"
    return True, None


def make_diagram_content(llm: LlamaCpp):
    def diagram_content(description: str, diagram_type: str = "flowchart") -> ContentResult:
        cfg = DIAGRAM_CONFIGS[diagram_type]
        prompt = f"""<start_of_turn>user
Produce a Mermaid {diagram_type} for: "{description}"
Return ONLY raw Mermaid syntax — no markdown fences, no explanation, no numbered prose list, no markdown bold (**), no HTML tags like <br>. {cfg['instruction']} Example (about a different topic, showing the format only):
{cfg['example']}
<end_of_turn>
<start_of_turn>model
"""
        code = llm.invoke(prompt, max_tokens=300).strip()
        code = re.sub(r"```(?:mermaid)?\s*([\s\S]*?)\s*```", r"\1", code).strip()
        # The prompt wraps description in quotes ('for: "{description}"') and
        # the model sometimes echoes a stray quote onto either end of its own
        # output — observed directly: an otherwise perfectly valid diagram
        # line failed validation purely because of one trailing '"' breaking
        # the end-anchored regex match.
        code = code.strip('"').strip()
        if diagram_type == "flowchart":
            code = _quote_risky_flowchart_labels(code)
        valid, reason = validate_diagram(code, diagram_type)
        return {"type": "diagram", "path": None, "markup": code, "caption": description,
                 "source": "generated", "status": "pending" if valid else "rejected",
                 "validation_notes": reason}

    return diagram_content
