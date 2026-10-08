"""All LLM calls go through this module.

- Model: qwen3:8b served locally by Ollama.
- Structured output: Ollama is given the Pydantic model's JSON schema (`format=`), and the reply
  is validated with Pydantic. Invalid replies are retried, then reported as failures (never guessed).
- Reproducibility: temperature 0, fixed seed, and every reply is cached in cache/llm_responses.json,
  keyed by a hash of (model, system prompt, prompt, schema). Re-running a phase costs zero LLM calls.
- Accounting: STATS counts real calls, cache hits, failures and generation time, so every phase
  can report how much LLM it used.
"""
import hashlib
import json
import time
from pathlib import Path

from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "cache" / "llm_responses.json"
MODEL = "qwen3:8b"
SEED = 42

STATS = {"calls": 0, "cache_hits": 0, "failures": 0, "seconds": 0.0, "output_tokens": 0,
         "oom_retries": 0}

_cache: dict | None = None


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    return _cache


def _save_cache() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(_cache, ensure_ascii=False, indent=1), encoding="utf-8")


def _key(system: str, prompt: str, schema: type[BaseModel], model: str) -> str:
    raw = json.dumps([model, system, prompt, schema.__name__, schema.model_json_schema()], sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


NUM_CTX = 2048  # our prompts are < 1,000 tokens; a 4,096 context needs 0.4 GB more RAM than this laptop has


def _chat(model: str, messages: list, schema: type[BaseModel]):
    """One Ollama call, with two out-of-memory fallbacks for a 7.4 GB RAM / 4 GB VRAM laptop:
    - GPU out of memory (PyTorch models such as SBERT or GLiNER hold VRAM): put fewer layers on the GPU
    - CPU out of memory (other programs hold RAM): use a smaller context window"""
    import ollama  # imported here so modules that never call the LLM do not need Ollama running
    # num_predict caps the reply length: without it a looping model writes until the context is full (~8 min)
    options = {"temperature": 0, "seed": SEED, "num_ctx": NUM_CTX, "num_predict": 300}
    for _ in range(3):
        try:
            return ollama.chat(model=model, messages=messages, format=schema.model_json_schema(),
                               think=False,  # qwen3's reasoning mode multiplies tokens; not needed
                               options=options)
        except ollama.ResponseError as e:
            msg = str(e)
            if "out of memory" not in msg:
                raise
            STATS["oom_retries"] += 1
            if "cuda" in msg.lower():
                options["num_gpu"] = 12 if "num_gpu" not in options else 0
            else:
                options["num_ctx"] = 1024
    raise RuntimeError("Ollama ran out of memory three times; close other programs and retry")


def ask_json(prompt: str, schema: type[BaseModel], system: str = "",
             model: str = MODEL, retries: int = 2) -> BaseModel | None:
    """Ask the LLM and return a validated `schema` instance, or None if every attempt was invalid."""

    cache = _load_cache()
    key = _key(system, prompt, schema, model)
    if key in cache:
        STATS["cache_hits"] += 1
        return schema.model_validate(cache[key]["response"])

    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    for _ in range(retries + 1):
        start = time.perf_counter()
        resp = _chat(model, messages, schema)
        STATS["calls"] += 1
        STATS["seconds"] += time.perf_counter() - start
        STATS["output_tokens"] += resp.get("eval_count", 0) or 0
        try:
            parsed = schema.model_validate_json(resp["message"]["content"])
        except ValidationError:
            continue
        cache[key] = {"model": model, "prompt": prompt, "response": parsed.model_dump()}
        _save_cache()
        return parsed

    STATS["failures"] += 1
    return None


def unload(model: str = MODEL) -> None:
    """Free the model from RAM/VRAM right away (Ollama otherwise keeps it for 5 minutes).
    With 7.4 GB of RAM, a loaded qwen3:8b leaves ~0.2 GB free, which is not enough to load
    numba/UMAP or SBERT safely, so every LLM stage ends with unload()."""
    import ollama
    ollama.generate(model=model, prompt="", keep_alive=0)


def report() -> str:
    s = STATS
    rate = s["output_tokens"] / s["seconds"] if s["seconds"] else 0
    return (f"LLM calls={s['calls']}  cache hits={s['cache_hits']}  failures={s['failures']}  "
            f"time={s['seconds']:.1f}s  output tokens={s['output_tokens']}  ({rate:.1f} tok/s)")
