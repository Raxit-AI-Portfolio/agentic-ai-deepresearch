"""LangChain-wrapped services (LLM, search, vision checker) plus small
stdlib-only helpers (LLM JSON parsing, copyright-overlap heuristic, URL
linkification) shared across multiple nodes."""

from __future__ import annotations

import ast
import difflib
import json
import os
import random
import re
import time
from typing import Dict, List, Union

from langchain_community.llms import LlamaCpp
from langchain_community.utilities import GoogleSerperAPIWrapper

from .config import ResearchConfig

# Supported values for ResearchConfig.llm_provider.
LLM_PROVIDER_GEMMA_7B = "gemma_7b"
LLM_PROVIDER_GEMINI_FLASH_LITE = "gemini-2.5-flash-lite"

# Exception class names the Gemini SDK (via google-api-core) raises for
# transient failures — rate-limit (429), server overload/unavailable (503),
# and timeout (504). Matched by name rather than importing google.api_core.
# exceptions directly so this module doesn't hard-require that package if
# it's ever absent; the message-substring fallback in
# _is_retryable_gemini_error covers the same cases either way.
_RETRYABLE_GEMINI_EXCEPTION_NAMES = (
    "ResourceExhausted", "TooManyRequests", "ServiceUnavailable",
    "DeadlineExceeded", "InternalServerError",
)


def _is_retryable_gemini_error(exc: Exception) -> bool:
    """True for a transient Gemini API error worth retrying (rate-limit,
    server overload, timeout) — false for a genuine bad-request/auth error
    (400/401/403), which would just fail the same way again."""
    if type(exc).__name__ in _RETRYABLE_GEMINI_EXCEPTION_NAMES:
        return True
    message = str(exc).lower()
    return any(token in message for token in (
        "429", "resource_exhausted", "rate limit", "rate_limit", "quota",
        "503", "service unavailable", "overloaded", "deadline exceeded",
    ))


def _call_with_retry(fn, *, max_attempts: int = 5, base_delay: float = 2.0, max_delay: float = 60.0):
    """Retries `fn()` on a transient Gemini error with exponential backoff +
    jitter. Needed because the free tier's 10 requests/minute cap makes an
    occasional 429 an expected part of a normal run, not an exceptional
    failure, and langchain-google-genai doesn't retry these on its own. A
    non-retryable error (bad prompt, auth failure) is re-raised immediately
    on its first occurrence."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            attempt += 1
            if attempt >= max_attempts or not _is_retryable_gemini_error(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1))) + random.uniform(0, 1)
            print(f"services: Gemini call failed ({exc!r}) — retrying in {delay:.1f}s "
                  f"(attempt {attempt}/{max_attempts})")
            time.sleep(delay)


def validate_model_file(model_path: str) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    size_gb = os.path.getsize(model_path) / (1024 ** 3)
    if size_gb < 1.0:
        raise ValueError(f"Model file is too small ({size_gb:.1f} GB) — likely incomplete.")


class GeminiLLM:
    """Wraps ChatGoogleGenerativeAI behind the same minimal interface the rest
    of the codebase already uses for the local LlamaCpp LLM —
    `.invoke(prompt, max_tokens=...) -> str` — so nodes_plan.py/
    nodes_research.py/content.py can call `llm.invoke(...)` and get a plain
    string back regardless of which provider is configured."""

    def __init__(self, model_name: str, temperature: float, top_p: float):
        from langchain_google_genai import ChatGoogleGenerativeAI

        self._chat = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, top_p=top_p)

    def invoke(self, prompt: str, max_tokens: int | None = None, **kwargs) -> str:
        if max_tokens is not None:
            kwargs.setdefault("max_output_tokens", max_tokens)
        return _call_with_retry(lambda: self._chat.invoke(prompt, **kwargs).content)


def build_llm(config: ResearchConfig) -> Union[LlamaCpp, GeminiLLM]:
    provider = (config.llm_provider or LLM_PROVIDER_GEMMA_7B).strip().lower()

    if provider == LLM_PROVIDER_GEMINI_FLASH_LITE:
        return GeminiLLM(
            model_name=LLM_PROVIDER_GEMINI_FLASH_LITE,
            temperature=config.temperature,
            top_p=config.top_p,
        )

    if provider != LLM_PROVIDER_GEMMA_7B:
        raise ValueError(
            f"Unknown llm_provider {config.llm_provider!r}; expected "
            f"{LLM_PROVIDER_GEMMA_7B!r} or {LLM_PROVIDER_GEMINI_FLASH_LITE!r}."
        )

    validate_model_file(config.model_path)
    return LlamaCpp(
        model_path=config.model_path,
        n_ctx=config.context_length,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        n_gpu_layers=-1,
        use_mmap=True,
        use_mlock=False,
        n_batch=512,
        stop=["<end_of_turn>", "<eos>"],
        verbose=False,
    )


def build_search_tool(config: ResearchConfig) -> GoogleSerperAPIWrapper:
    return GoogleSerperAPIWrapper(serper_api_key=config.search_api_key, k=5)


def noop_vision_checker(path: str) -> str:
    """Stand-in for make_gemini_vision_checker's check() when no
    GOOGLE_API_KEY/GEMINI_API_KEY is configured — accepts every image
    without a vision-model call. validate_node currently never invokes the
    vision checker at all (see its "Vision-model check bypassed on request"
    comment in nodes_research.py), so this only matters if that check is
    ever re-enabled without an API key configured — in that case media
    passes through unchecked rather than the whole graph failing to build.
    """
    return "OK"


def make_gemini_vision_checker(model_name: str):
    """Lazy import so this module doesn't hard-require langchain-google-genai
    unless the validator actually needs it."""
    from langchain_core.messages import HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
    import base64

    vision_llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    def check(path: str) -> str:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        message = HumanMessage(content=[
            {"type": "text", "text": (
                "Assess this image for (1) safe-for-work content and (2) any "
                "recognizable copyrighted character/logo/brand or identifiable "
                "real person. Reply with exactly 'OK' if clear on both, "
                "otherwise a one-sentence reason it should be rejected."
            )},
            {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
        ])
        return _call_with_retry(lambda: vision_llm.invoke([message]).content.strip())

    return check


# Small instruct models frequently typeset JSON with "smart" typographic
# punctuation instead of the plain ASCII JSON requires. json.loads rejects
# all of these outright, so they're normalized away before parsing.
_SMART_PUNCTUATION = {
    "“": '"', "”": '"',   # left/right double quotation mark
    "‘": "'", "’": "'",   # left/right single quotation mark
    " ": " ", " ": " ",   # line/paragraph separator
    " ": " ",                     # non-breaking space
}


def _normalize_llm_json_text(text: str) -> str:
    for bad, good in _SMART_PUNCTUATION.items():
        text = text.replace(bad, good)
    return re.sub(r",\s*([\]}])", r"\1", text)  # drop trailing commas


def parse_llm_json(response_text: str):
    if not response_text:
        return None
    clean = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", response_text).strip()
    clean = _normalize_llm_json_text(clean)

    candidates = [clean]
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", clean)
    if match:
        candidates.append(match.group(1))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # After normalization, mixed single/double-quoted output (which
        # json.loads always rejects) is usually valid Python literal syntax
        # — e.g. a model writing 'foo' instead of "foo" for a string value.
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, (list, dict)):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return None


def _max_overlap_ratio(text: str, context: str) -> float:
    """Rough copyright signal (stdlib-only): longest common substring between
    the generated section and its raw source snippets, as a share of the
    section's own length."""
    if not text or not context:
        return 0.0
    matcher = difflib.SequenceMatcher(None, text.lower(), context.lower())
    match = matcher.find_longest_match(0, len(text), 0, len(context))
    return match.size / max(len(text), 1)


_PREAMBLE_RE = re.compile(
    r"^\s*(?:sure|certainly|of course|okay|alright|understood|here)\b[^:\n]{0,140}:\s*",
    re.IGNORECASE,
)


def strip_llm_preamble(text: str) -> str:
    """Removes a leading conversational lead-in ('Sure, here are the new
    sentences about "X": ...', 'Here's the paragraph: ...') that a small
    instruct model sometimes prepends to what should be pure prose output —
    observed directly leaking into a report section verbatim. Only strips a
    clause matching this shape at the very start of the text and only up to
    its first colon; text without this pattern is returned unchanged."""
    return _PREAMBLE_RE.sub("", text or "", count=1).strip()


# Narrow, high-confidence subset of the "missing space between glued words"
# bug (open-issues.md) — only the two glue shapes that can be fixed without
# risking a false positive on a real word/brand name:
#   1. a closing parenthesis running directly into the next letter
#      ("(OVs)with" -> "(OVs) with") — unambiguous, a ")" is never meant to
#      touch the next word.
#   2. one of a small safelist of common connector words running directly
#      into a following Title-Case or ALL-CAPS word ("withCAR" -> "with
#      CAR", "bothBoeing" -> "both Boeing") — deliberately whole-word only
#      (via \b) so this never touches a real camelCase brand name like
#      "iPhone"/"eBay"/"McDonald's", none of which start with a safelisted
#      connector word.
# Doesn't catch every case from open-issues.md (e.g. "resultedin",
# "carriersto" — two lowercase words glued with no case change at all)
# since telling that apart from a genuine compound word needs a real
# dictionary, which isn't worth adding for this; left as a known gap, same
# as this codebase's other partial-mitigation fixes.
_PAREN_GLUE_RE = re.compile(r"(?<=\))(?=[A-Za-z])")
_CONNECTOR_GLUE_RE = re.compile(
    r"\b(both|with|using|and|as|for|in|on|to|from|between|among|across|within|before|after|during|via|by)"
    r"(?=[A-Z])"
)


def fix_missing_spaces(text: str) -> str:
    """Inserts a space at the two glue shapes described above. See
    _PAREN_GLUE_RE/_CONNECTOR_GLUE_RE for exactly what is and isn't covered."""
    text = _PAREN_GLUE_RE.sub(" ", text)
    text = _CONNECTOR_GLUE_RE.sub(lambda m: f"{m.group(1)} ", text)
    return text


def split_sentences(text: str) -> List[str]:
    """Splits on sentence-ending punctuation followed by whitespace — good
    enough for counting/filtering LLM-generated prose; no need for a real
    NLP sentence tokenizer at this scale."""
    return [s for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s]


def drop_duplicate_sentences(text: str, context: str, threshold: float = 0.6,
                              first_sentence_threshold: float = 0.45) -> str:
    """Removes any sentence from `text` that's substantially the same as
    something already in `context`, checked per sentence via the same
    longest-common-substring ratio as _max_overlap_ratio. Used in place of
    asking the model not to repeat itself — unreliable with a small
    quantized model (see nodes_research.py/GRAPH.md) — to enforce
    non-duplication deterministically instead: strip what's actually
    duplicate, keep what's genuinely new.

    The first sentence gets a stricter (lower) threshold than the rest: this
    model has a demonstrated tendency to open different sections with the
    same templated framing ("The recent string of ... has cast a spotlight
    on ...") worded just differently enough each time to land under the
    normal threshold (observed directly at ~0.54, under the 0.6 default) —
    while genuinely distinct sections rarely share that much of their
    opening clause specifically. Applying the stricter bar everywhere would
    risk stripping ordinary topical vocabulary overlap in body sentences,
    which is expected and fine."""
    sentences = split_sentences(text)
    if not context:
        return " ".join(sentences)
    kept = []
    for i, s in enumerate(sentences):
        limit = first_sentence_threshold if i == 0 else threshold
        if _max_overlap_ratio(s, context) < limit:
            kept.append(s)
    return " ".join(kept)


def drop_self_duplicate_sentences(text: str, literal_threshold: float = 0.6,
                                   semantic_threshold: float = 0.2) -> str:
    """Removes any sentence that's substantially the same as an EARLIER
    sentence within the same text — catches a model restating its own claim
    twice in one paragraph, which drop_duplicate_sentences can't catch since
    it only ever compares against OTHER sections, never a section against
    itself.

    Two checks, each candidate compared against every already-kept sentence
    INDIVIDUALLY (never against them all joined into one blob) — same
    max-over-existing pattern as filter_novel_pointers, and for the same
    reason: joining every kept sentence into one bag-of-words dilutes a
    real duplicate's signal below that of merely-related sentences (verified
    directly — a real duplicate scored LOWER than distinct sentences when
    compared to the joined blob, and clearly highest when compared
    per-sentence instead).
    - Literal: longest-common-substring ratio, same as elsewhere — catches
      near-verbatim restatement (observed directly: "poor positioning of
      engines and single sensor malfunction" stated twice in one section).
    - Semantic: TF-IDF + cosine — catches the SAME claim in different words
      (observed directly: "demonstrating high levels safety and efficacy"
      restated later as "demonstrate high levels safety and efficacy" about
      a different subject), which the literal check alone missed. The
      threshold is much lower than filter_novel_pointers' 0.5 because full
      sentences carry far more incidental vocabulary than short 5-8 word
      pointer phrases, which dilutes cosine similarity even for a genuine
      duplicate — calibrated against real duplicate vs. distinct sentence
      pairs pulled from an actual generated section (duplicate: ~0.26;
      closest genuinely-distinct sentence: ~0.13)."""
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return " ".join(sentences)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    kept: List[str] = [sentences[0]]
    for s in sentences[1:]:
        max_literal = max((_max_overlap_ratio(s, k) for k in kept), default=0.0)
        try:
            matrix = TfidfVectorizer(stop_words="english").fit_transform(kept + [s])
            sims = cosine_similarity(matrix[-1:], matrix[:-1])[0]
            max_semantic = max(sims) if len(sims) else 0.0
        except ValueError:
            max_semantic = 0.0  # empty vocabulary after stop-word removal — nothing to score
        if max_literal < literal_threshold and max_semantic < semantic_threshold:
            kept.append(s)
    return " ".join(kept)


def filter_novel_pointers(candidates: List[str], existing: List[str],
                           literal_threshold: float = 0.6, semantic_threshold: float = 0.5) -> List[str]:
    """Keeps only the candidate fact-pointers (short phrases extracted from a
    section's Research Notes, see make_write_node) that aren't already
    covered by `existing` pointers from previously-written sections. Two
    checks, both reused from elsewhere in the codebase rather than adding
    anything new: literal near-duplicate text via the same longest-common-
    substring ratio used for the copyright/duplication overlap checks, and
    semantic similarity via TF-IDF + cosine — the same embedding-free
    approach already used for reference-doc matching (match_reference_docs)
    and appropriate here for the same reason: a handful of short phrases,
    compared once, no persistent index needed."""
    if not candidates:
        return []
    if not existing:
        return candidates

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    matrix = TfidfVectorizer(stop_words="english").fit_transform(existing + candidates)
    existing_matrix = matrix[:len(existing)]

    novel = []
    for i, candidate in enumerate(candidates):
        max_literal = max((_max_overlap_ratio(candidate, e) for e in existing), default=0.0)
        sims = cosine_similarity(matrix[len(existing) + i], existing_matrix)[0]
        max_semantic = max(sims) if len(sims) else 0.0
        if max_literal < literal_threshold and max_semantic < semantic_threshold:
            novel.append(candidate)
    return novel


def filter_relevant_results(query: str, results: List[dict], threshold: float = 0.05) -> List[dict]:
    """Drops search results that are essentially unrelated to the query that
    fetched them — TF-IDF + cosine similarity between the query and each
    result's title+snippet, the same embedding-free approach already used
    for reference-doc matching (reference_docs.py::match_reference_docs).
    Serper sometimes returns results with almost no real connection to the
    query (an unrelated company's annual report, a decades-old PDF) that the
    model then dutifully but nonsensically cites — observed directly: a
    German car-market report citing a 1944 "Survey of Current Business" PDF
    and an unrelated company's 10-K filing.

    Never returns an empty list when `results` was non-empty: if nothing
    clears the bar (e.g. the query and results share no real vocabulary at
    all — TF-IDF can't score that meaningfully either way), it's safer to
    keep every result than to leave a section with zero sources."""
    if not results:
        return results

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [f"{r.get('title', '')} {r.get('snippet', '')}" for r in results]
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform([query] + texts)
    except ValueError:
        return results  # empty vocabulary after stop-word removal — nothing to score
    sims = cosine_similarity(matrix[0], matrix[1:])[0]
    kept = [r for r, sim in zip(results, sims) if sim >= threshold]
    return kept or results


def _linkify_urls(text: str, citation_map: Dict[str, int]) -> str:
    def repl(m):
        num = citation_map.get(m.group(0))
        return f"[{num}]" if num else m.group(0)
    return re.sub(r"https?://\S+", repl, text)
