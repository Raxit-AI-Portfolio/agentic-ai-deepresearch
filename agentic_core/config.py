"""Config (reused from the notebook; ReportConfig bug #3 fixed):
ReportConfig.get_standard_themes() now actually receives report_format
instead of referencing an undefined name."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


@dataclass
class ResearchConfig:
    # Which LLM backs the pipeline's text generation (plan/write/validate/
    # content-spec nodes). services.py::build_llm branches on this value:
    #
    #   "gemma_7b" (default) — local GGUF model via llama-cpp-python.
    #     - Needs `model_path` below pointing at an actual .gguf file on disk
    #       (validated by services.validate_model_file — must exist, >1GB).
    #     - No API key / network call; runs on this machine (GPU via
    #       CMAKE_ARGS="-DGGML_CUDA=on" at install time, or CPU — slower).
    #     - Usage: ResearchConfig()  # or explicitly llm_provider="gemma_7b"
    #
    #   "gemini-2.5-flash-lite" — hosted Gemini via langchain-google-genai.
    #     - `model_path` is ignored for this provider.
    #     - Needs GOOGLE_API_KEY set in the environment/.env (same variable
    #       ChatGoogleGenerativeAI already reads for `vision_model` below).
    #     - Usage: ResearchConfig(llm_provider="gemini-2.5-flash-lite")
    #
    # Env-var override (mirrors model_path below) so the same code can run
    # gemini locally and gemma_7b in Colab (or vice versa) without editing
    # this default per environment: set DEEP_RESEARCH_LLM_PROVIDER.
    llm_provider: str = os.getenv("DEEP_RESEARCH_LLM_PROVIDER", "gemini-2.5-flash-lite")
    # Relative default matches the repo-local layout tests/scripts run
    # against. Colab (and any other environment where the .gguf lives
    # outside the repo, e.g. a separate Drive folder) should set
    # DEEP_RESEARCH_MODEL_PATH instead of editing this default.
    model_path: str = os.getenv("DEEP_RESEARCH_MODEL_PATH", "models/gemma-7b-it-Q4_K_M.gguf")
    max_tokens: int = 2048
    temperature: float = 0.4
    top_p: float = 0.95
    top_k: int = 40
    context_length: int = 4096
    search_api_key: str = os.getenv("SERPER_API_KEY", "")
    vision_model: str = "gemini-3.6-flash"


class ReportConfig:
    def get_meta_themes(self) -> List[str]:
        return ["EXECUTIVE SUMMARY", "ABSTRACT", "INTRODUCTION",
                "OVERVIEW", "CONCLUSION", "NEXT STEPS"]

    def get_standard_themes(self, report_format: str) -> List[str]:
        meta_themes = self.get_meta_themes()
        return [t for t in self.themes_for_format(report_format) if t not in meta_themes]

    def themes_for_format(self, fmt: str) -> List[str]:
        fmt = (fmt or "").strip().lower()
        if fmt == "executive":
            return ["EXECUTIVE SUMMARY"]
        if fmt == "detailed":
            return ["INTRODUCTION", "DETAILED ANALYSIS",
                     "CURRENT TRENDS AND DEVELOPMENTS",
                     "IMPLICATIONS AND RECOMMENDATIONS", "CONCLUSION"]
        if fmt == "academic":
            return ["ABSTRACT", "INTRODUCTION", "METHODOLOGY",
                     "FINDINGS", "DISCUSSION", "CONCLUSION"]
        if fmt == "presentation":
            return ["OVERVIEW", "KEY INSIGHTS", "RECOMMENDATIONS",
                     "NEXT STEPS", "CONCLUSION"]
        return ["INTRODUCTION", "DETAILED ANALYSIS", "CONCLUSION"]
