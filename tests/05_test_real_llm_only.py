"""
Test 5: first contact with the REAL model — confirms the gguf file loads and
responds before you trust it inside the full graph.
Needs: the actual .gguf file at ResearchConfig.model_path (default
"models/gemma-7b-it-Q4_K_M.gguf", relative to the project root — drop your
downloaded model there, or override via MODEL_PATH below),
GPU/llama-cpp-python built with GPU support (CPU-only also works, slower).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_core import ResearchConfig, build_llm

if __name__ == "__main__":
    print("Running real LLM smoke test...")
    # Override with an absolute path if your model file lives elsewhere.
    model_path = os.getenv("MODEL_PATH", ResearchConfig().model_path)
    print(f"Using model path: {model_path}")
    config = ResearchConfig(model_path=model_path)
    llm = build_llm(config)

    prompt = "<start_of_turn>user\nSay hello in exactly one sentence.<end_of_turn>\n<start_of_turn>model\n"
    response = llm.invoke(prompt)
    print("Model response:", response)

    assert response.strip(), "Model returned an empty response — check the model file/prompt template."
    print("\nReal LLM smoke test passed.")
