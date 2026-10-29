"""
Run this FIRST. Reports which packages are installed vs missing, grouped by
which part of the pipeline needs them — so you know what to `pip install`
before chasing a confusing ImportError three layers deep in the graph.
"""
import importlib

REQUIREMENTS = {
    "Core graph":        ["langgraph", "langchain_community", "dotenv"],
    "Reference docs":    ["pdfplumber", "docx", "bs4", "pytesseract", "PIL", "fitz"],
    "Non-text content":  ["sklearn", "pandas", "matplotlib", "requests", "tabulate"],
    "Vision validator":  ["langchain_google_genai", "langchain_core"],
    "Optional export":   ["markdown", "weasyprint", "pypandoc"],
}

PIP_NAMES = {
    "dotenv": "python-dotenv", "docx": "python-docx", "bs4": "beautifulsoup4",
    "PIL": "Pillow", "fitz": "PyMuPDF", "sklearn": "scikit-learn",
}

if __name__ == "__main__":
    missing_overall = []
    for group, modules in REQUIREMENTS.items():
        print(f"\n--- {group} ---")
        for mod in modules:
            try:
                importlib.import_module(mod)
                print(f"  OK       {mod}")
            except ImportError:
                pip_name = PIP_NAMES.get(mod, mod)
                print(f"  MISSING  {mod}  ->  pip install {pip_name}")
                missing_overall.append(pip_name)

    print("\n" + "=" * 60)
    if missing_overall:
        print("Install the missing ones with:")
        print(f"  pip install {' '.join(sorted(set(missing_overall)))}")
    else:
        print("All dependencies present. Optional-export ones only matter if")
        print("you actually request html/pdf/document output formats.")
