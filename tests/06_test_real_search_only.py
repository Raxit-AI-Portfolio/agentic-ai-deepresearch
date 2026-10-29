"""
Test 6: confirms your Serper API key works before trusting it inside the
graph. Needs: SERPER_API_KEY set in your .env file.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_core import ResearchConfig, build_search_tool

if __name__ == "__main__":
    config = ResearchConfig()  # reads SERPER_API_KEY from .env via load_dotenv()
    assert config.search_api_key, "SERPER_API_KEY not found — check your .env file."

    search = build_search_tool(config)
    data = search.results("cheese manufacturing Italy")

    organic = data.get("organic", [])
    print("Top result:", organic[0] if organic else "No results — check your API key/quota.")
    assert organic, "Expected at least one organic result."
    print("\nReal search smoke test passed.")
