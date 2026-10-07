"""Phase 0: check that the toolchain works on this machine and measure LLM speed.

    python scripts/check_env.py

1. Times qwen3:8b on 5 real attribute-extraction prompts (structured JSON output).
   This runs FIRST, before PyTorch touches the GPU: the 4 GB card cannot hold a PyTorch CUDA
   context and Ollama's share of the 8B model at the same time.
2. Prints library versions, whether PyTorch sees the GPU, and a spaCy smoke test.
"""
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ollama
from pydantic import BaseModel

from src import llm

# ---------- 1. LLM speed test ----------


class Attributes(BaseModel):
    brand: str | None
    product_type: str | None
    quantity: float | None
    unit: str | None


names = ["Fortune Sunlite Refined Sunflower Oil 1 ltr Pouch",
         "Tata Salt Lite Low Sodium 1 kg",
         "Britannia Good Day Cashew Cookies 200 gm",
         "Amul Pure Ghee 500 ml Tin",
         "Maggi 2-Minute Masala Noodles pack of 4"]

# The benchmark must measure real generation, so it skips the cache.
llm._cache = {}
llm._save_cache = lambda: None

t0 = time.perf_counter()
llm.ask_json("Say hi.", Attributes)  # loads the model into memory
print(f"model load + first call: {time.perf_counter() - t0:.1f}s")
for m in ollama.ps().models:
    print(f"model in memory: {m.model}  size={m.size / 2**30:.1f} GiB  on GPU={m.size_vram / 2**30:.1f} GiB "
          f"({100 * m.size_vram / m.size:.0f}%)")

llm.STATS.update(calls=0, cache_hits=0, failures=0, seconds=0.0, output_tokens=0)
for n in names:
    out = llm.ask_json(f"Extract the attributes of this product name: {n}", Attributes,
                       system="You extract product attributes. Answer only with JSON.")
    print(f"  {n!r:55s} -> {out.model_dump() if out else 'INVALID'}")
print(llm.report())
per = llm.STATS["seconds"] / max(llm.STATS["calls"], 1)
print(f"seconds per call: {per:.2f}  ->  2,000 names would take ~{2000 * per / 3600:.1f} h")
llm.unload()  # frees ~5.6 GB; with the model loaded only ~0.2 GB of RAM is left for Python

# ---------- 2. Libraries, GPU, spaCy ----------
print()
for pkg in ["pandas", "spacy", "sklearn", "sentence_transformers", "faiss", "umap", "gliner",
            "rapidfuzz", "symspellpy", "ollama", "pydantic", "bertopic", "streamlit", "gensim",
            "duckdb", "torch"]:
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg:22s} {getattr(m, '__version__', 'ok')}")
    except Exception as e:
        print(f"{pkg:22s} MISSING ({e.__class__.__name__}: {e})")

import torch

print("CUDA available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")

import spacy

nlp = spacy.load("en_core_web_sm")
text = "Britannia Good Day Cashew Biscuits 200 gm"
print("spaCy lemmas (original case):", [t.lemma_ for t in nlp(text)])
print("spaCy lemmas (lowercased):   ", [t.lemma_ for t in nlp(text.lower())])
