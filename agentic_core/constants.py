"""Module-wide constants shared across the graph's nodes."""

import os
import tempfile

MAX_REVISIONS = 2
COPYRIGHT_OVERLAP_THRESHOLD = 0.35
# tempfile.gettempdir() rather than a hardcoded "/tmp" — the latter is a
# real, pre-existing directory on Linux/Colab but not on Windows (and can
# fail outright at the root of a virtual/cloud-sync drive, e.g. a Google
# Drive letter, where creating a folder at the drive root isn't permitted).
DATA_DIR = os.getenv("DEEP_RESEARCH_DATA_DIR") or os.path.join(tempfile.gettempdir(), "deep_research_runs")

# Per theme, at most this many of its highest-weighted sub-topics are asked
# whether they'd benefit from non-text content — see make_content_requests_node.
# Not a target to hit: the LLM is free to decline all of them.
MAX_CONTENT_REQUESTS_PER_THEME = 3
