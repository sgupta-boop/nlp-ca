"""Phase 1: column standardization (R1) and column selection (R2).

Three ways to map a source column to the canonical schema:
  1. rules       - keyword rules on the header name only
  2. embeddings  - Sentence-BERT: embed "header + 5 sample values", pick the canonical field
                   whose description is most cosine-similar
  3. emb + LLM   - as 2, but when the best cosine is below a threshold, ask the LLM (JSON output)

Column selection: profile every text column (token count, uniqueness, alphabetic ratio, entropy)
and keep the free-text, high-uniqueness ones as clustering candidates, with a written reason.

Run:  python -m src.columns      (evaluates on data/gold/header_mapping.csv, writes canonical tables)
"""
import math
import re
from collections import Counter
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src import data_io, llm

ROOT = data_io.ROOT
SEED = 42

# The canonical schema (roadmap) plus "other" for columns that have no slot in it.
# The descriptions are what the embedding model compares each column against.
FIELD_DESCRIPTIONS = {
    "product_name": "product name or title of the item, e.g. Amul Butter 500 g, Sony Turntable PSLX350H",
    "brand": "brand or manufacturer of the product, e.g. Amul, Sony, Nestle",
    "category": "product category or department, e.g. Beverages, Electronics, Clothing",
    "sub_category": "product sub-category, a finer category inside a category, e.g. Tea, Hair Care",
    "price": "selling price or discounted price actually charged, a number like 49.5",
    "mrp": "maximum retail price, list price or market price before discount, a number",
    "quantity": "pack size or net quantity with unit, e.g. 500 g, 1 l, 6 pcs",
    "description": "long free-text description of the product and its features",
    "rating": "customer rating or review score, e.g. 4.2 out of 5",
    "id": "unique identifier, code, SKU or barcode of the record",
    "other": "other metadata such as url, image link, timestamp, creator, packaging, country, "
             "labels, nutrition grade, ingredients or specifications",
}
FIELDS = list(FIELD_DESCRIPTIONS)
CANONICAL = [f for f in FIELDS if f != "other"]

# ---------- 1. rule baseline ----------
# Generic e-commerce header keywords, checked in this order (specific before general).
KEYWORD_RULES = [
    ("other", {"url", "image", "img", "timestamp", "date", "time", "created", "modified", "link"}),
    ("sub_category", {"subcategory", "sub"}),
    ("mrp", {"mrp", "msrp", "list"}),
    ("price", {"price", "cost"}),
    ("rating", {"rating", "stars", "review"}),
    ("description", {"description", "desc", "details"}),
    ("brand", {"brand", "brands", "manufacturer", "make"}),
    ("category", {"category", "categories", "department"}),
    ("quantity", {"quantity", "qty", "weight", "volume", "size"}),
    ("product_name", {"name", "title"}),
    ("id", {"id", "sku", "code", "uuid"}),
]


def header_tokens(header: str) -> list[str]:
    """'product_category_tree' -> ['product', 'category', 'tree']; 'isFKAdvantage' is split on case."""
    h = re.sub(r"([a-z])([A-Z])", r"\1 \2", header)
    return [t for t in re.split(r"[^a-zA-Z0-9]+", h.lower()) if t]


def map_by_rules(header: str) -> str:
    tokens = set(header_tokens(header))
    for field, keywords in KEYWORD_RULES:
        if tokens & keywords:
            return field
    return "other"


# ---------- 2. embeddings ----------

def sample_values(series: pd.Series, n: int = 5, max_chars: int = 60) -> list[str]:
    """n distinct non-empty example values, shortened, chosen with a fixed seed."""
    vals = series.dropna().astype(str).str.strip()
    vals = vals[vals != ""].drop_duplicates()
    if len(vals) > n:
        vals = vals.sample(n, random_state=SEED)
    return [v[:max_chars] for v in vals]


def column_text(header: str, samples: list[str]) -> str:
    return f"column {' '.join(header_tokens(header))}: values " + "; ".join(samples)


_model = None


def get_model(name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(name)
    return _model


def embedding_scores(headers: list[str], samples: list[list[str]], use_samples: bool = True) -> np.ndarray:
    """Cosine similarity matrix: (n_columns x n_fields)."""
    model = get_model()
    texts = [column_text(h, s if use_samples else []) for h, s in zip(headers, samples)]
    fields = [f"{f.replace('_', ' ')}: {d}" for f, d in FIELD_DESCRIPTIONS.items()]
    a = model.encode(texts, normalize_embeddings=True)
    b = model.encode(fields, normalize_embeddings=True)
    return a @ b.T


# ---------- 3. LLM fallback ----------

class FieldChoice(BaseModel):
    field: Literal["product_name", "brand", "category", "sub_category", "price", "mrp",
                   "quantity", "description", "rating", "id", "other"]


LLM_SYSTEM = "You map columns of product tables to a fixed schema. Answer only with JSON."


def map_by_llm(header: str, samples: list[str]) -> str | None:
    field_list = "\n".join(f"- {f}: {d}" for f, d in FIELD_DESCRIPTIONS.items())
    prompt = (f"Schema fields:\n{field_list}\n\n"
              f"Column header: {header}\nSample values: {samples}\n\n"
              "Which schema field does this column hold? Use 'other' if none fits.")
    out = llm.ask_json(prompt, FieldChoice, system=LLM_SYSTEM)
    return out.field if out else None


def rule_hit(header: str) -> bool:
    """True if some keyword rule fired (rather than the default 'other')."""
    tokens = set(header_tokens(header))
    return any(tokens & kw for _, kw in KEYWORD_RULES)


def map_columns(headers, samples, method: str = "cascade", threshold: float = 0.6):
    """The mapping used by the system.
    cascade:        keyword rules when a rule fires; otherwise the LLM (rules again if the LLM fails)
    embedding_llm:  roadmap method - embeddings (header + samples), LLM when best cosine < threshold
    """
    rows = []
    scores = embedding_scores(headers, samples) if method == "embedding_llm" else None
    for i, h in enumerate(headers):
        if method == "cascade":
            if rule_hit(h):
                field, decided = map_by_rules(h), "rules"
            else:
                field, decided = (map_by_llm(h, samples[i]) or map_by_rules(h)), "llm"
        else:
            best = int(scores[i].argmax())
            field, decided = FIELDS[best], "embedding"
            if scores[i, best] < threshold:
                field, decided = (map_by_llm(h, samples[i]) or field), "llm"
        rows.append({"header": h, "predicted": field, "decided_by": decided})
    return pd.DataFrame(rows)


# ---------- column profiling and selection (R2) ----------

def entropy(values: pd.Series) -> float:
    """Shannon entropy (bits) of the value distribution."""
    counts = values.value_counts(normalize=True)
    return float(-(counts * np.log2(counts)).sum())


def profile_column(s: pd.Series) -> dict:
    vals = s.dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    if len(vals) == 0:
        return {"non_null": 0.0, "avg_tokens": 0.0, "unique_ratio": 0.0, "alpha_ratio": 0.0,
                "entropy": 0.0, "url_ratio": 0.0}
    joined = "".join(vals.head(2000))
    return {
        "non_null": round(len(vals) / len(s), 3),
        "avg_tokens": round(vals.str.split().str.len().mean(), 2),
        "unique_ratio": round(vals.nunique() / len(vals), 3),
        "alpha_ratio": round(sum(c.isalpha() for c in joined) / max(len(joined), 1), 3),
        "entropy": round(entropy(vals), 2),
        "url_ratio": round(vals.str.contains(r"https?://|www\.", regex=True).mean(), 3),
    }


# thresholds are part of the method and are reported with the results
SELECT_RULES = {"min_avg_tokens": 2.0, "min_unique_ratio": 0.3, "min_alpha_ratio": 0.6,
                "max_url_ratio": 0.1, "min_non_null": 0.5}


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Profile every column and decide whether it is a clustering candidate, with the reason."""
    rows = []
    r = SELECT_RULES
    for col in df.columns:
        p = profile_column(df[col])
        reasons = []
        if p["non_null"] < r["min_non_null"]:
            reasons.append(f"mostly empty ({p['non_null']:.0%} filled)")
        if p["url_ratio"] > r["max_url_ratio"]:
            reasons.append("URLs")
        if p["alpha_ratio"] < r["min_alpha_ratio"]:
            reasons.append(f"mostly non-letters ({p['alpha_ratio']:.0%} letters: ids, prices, codes)")
        if p["avg_tokens"] < r["min_avg_tokens"]:
            reasons.append(f"too short ({p['avg_tokens']} tokens: a label, not free text)")
        if p["unique_ratio"] < r["min_unique_ratio"]:
            reasons.append(f"low uniqueness ({p['unique_ratio']:.0%}: a category-like column)")
        selected = not reasons
        reason = ("free text, " + f"{p['avg_tokens']} tokens, {p['unique_ratio']:.0%} unique, "
                  f"{p['alpha_ratio']:.0%} letters") if selected else "; ".join(reasons)
        rows.append({"column": col, **p, "selected": selected, "reason": reason})
    out = pd.DataFrame(rows)
    # the shortest selected text column is the name to cluster on; longer ones are context
    sel = out[out["selected"]]
    out["role"] = ""
    if len(sel):
        out.loc[sel["avg_tokens"].idxmin(), "role"] = "primary (name)"
        out.loc[sel.index.difference([sel["avg_tokens"].idxmin()]), "role"] = "secondary (context)"
    return out


def resolve_price_conflict(mapping: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Two columns both mapped to 'price' and none to 'mrp': the one with the higher median value
    is the MRP, because the maximum retail price is by definition never below the selling price."""
    mapping = mapping.copy()
    prices = mapping.index[mapping["predicted"] == "price"]
    if len(prices) == 2 and not (mapping["predicted"] == "mrp").any():
        medians = {i: to_number(df[mapping.at[i, "header"]]).median() for i in prices}
        mapping.at[max(medians, key=medians.get), "predicted"] = "mrp"
        mapping.loc[prices, "decided_by"] = mapping.loc[prices, "decided_by"] + " + price/mrp rule"
    return mapping


# ---------- building canonical tables ----------

def to_number(s: pd.Series) -> pd.Series:
    """First number in the text: 'Rs. 1,299.00' -> 1299.0, 'No rating available' -> NaN."""
    num = s.astype(str).str.extract(r"(\d[\d,]*(?:\.\d+)?)", expand=False).str.replace(",", "")
    return pd.to_numeric(num, errors="coerce")


def split_category_path(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Flipkart stores '["Clothing >> Women's Clothing >> ..."]'; return level 1 and level 2."""
    parts = s.astype(str).str.strip('[]"').str.split(r"\s*>>\s*")
    return parts.str[0].str.strip(), parts.str[1].str.strip()


def to_canonical(df: pd.DataFrame, mapping: pd.DataFrame, source: str) -> pd.DataFrame:
    """Rename mapped columns to canonical names. If several columns map to one field (e.g. Flipkart's
    product_rating and overall_rating), the first one in the table is kept. Fields with no source
    column are added as empty columns."""
    m = mapping[mapping["predicted"] != "other"].drop_duplicates("predicted")
    out = pd.DataFrame(index=df.index)
    for _, row in m.iterrows():
        out[row["predicted"]] = df[row["header"]]
    for f in CANONICAL:
        if f not in out:
            out[f] = pd.NA
    if out["category"].astype(str).str.contains(">>").any():
        cat, sub = split_category_path(out["category"])
        out["category"] = cat
        if out["sub_category"].isna().all():
            out["sub_category"] = sub
    for f in ("price", "mrp", "rating"):
        out[f] = to_number(out[f])
    out = out[CANONICAL]
    out.insert(0, "source", source)
    return out


# ---------- evaluation on the gold header set ----------

def load_sources() -> dict[str, pd.DataFrame]:
    ab, ag = data_io.load_abt_buy(), data_io.load_amazon_google()
    bb = data_io.load_bigbasket(keep_index=True)
    return {"bigbasket": bb, "flipkart": data_io.load_flipkart(),
            "abt": ab["left"], "buy": ab["right"], "amazon": ag["left"], "google": ag["right"],
            "off": data_io.load_off_sample()}


def gold_with_samples() -> pd.DataFrame:
    gold = pd.read_csv(ROOT / "data" / "gold" / "header_mapping.csv")
    sources = load_sources()
    gold["samples"] = [sample_values(sources[s][h]) for s, h in zip(gold["source"], gold["header"])]
    return gold


def evaluate(thresholds=(0.3, 0.4, 0.5, 0.6)) -> tuple[pd.DataFrame, pd.DataFrame]:
    gold = gold_with_samples()
    headers, samples = gold["header"].tolist(), gold["samples"].tolist()
    acc = lambda pred: round(float((pd.Series(pred, index=gold.index) == gold["gold_field"]).mean()), 3)

    results, preds = [], gold[["source", "header", "gold_field"]].copy()
    preds["rules"] = [map_by_rules(h) for h in headers]
    results.append({"method": "rules only (header keywords)", "accuracy": acc(preds["rules"]), "llm_calls": 0})

    for use_samples, label in ((False, "embeddings, header only"), (True, "embeddings, header + 5 samples")):
        s = embedding_scores(headers, samples, use_samples)
        col = "emb_header" if not use_samples else "embeddings"
        preds[col] = [FIELDS[i] for i in s.argmax(1)]
        preds[col + "_cos"] = s.max(1).round(3)
        results.append({"method": label, "accuracy": acc(preds[col]), "llm_calls": 0})

    llm.STATS.update(calls=0, cache_hits=0, failures=0, seconds=0.0, output_tokens=0)
    llm_all = [map_by_llm(h, s) for h, s in zip(headers, samples)]
    preds["llm_only"] = llm_all
    llm_report = llm.report()
    results.append({"method": "LLM only (qwen3:8b)", "accuracy": acc(llm_all), "llm_calls": len(headers)})

    for base, label in (("embeddings", "header + samples"), ("emb_header", "header only")):
        for t in thresholds:
            use = preds[base + "_cos"] < t
            hybrid = np.where(use, preds["llm_only"].fillna(preds[base]), preds[base])
            preds[f"{base}_llm_{t}"] = hybrid
            results.append({"method": f"embeddings ({label}) + LLM if cosine < {t}",
                            "accuracy": acc(hybrid), "llm_calls": int(use.sum())})

    hits = pd.Series([rule_hit(h) for h in headers], index=gold.index)
    preds["cascade"] = np.where(hits, preds["rules"], preds["llm_only"].fillna(preds["rules"]))
    results.append({"method": "CASCADE: rules if a keyword fires, else LLM",
                    "accuracy": acc(preds["cascade"]), "llm_calls": int((~hits).sum())})

    sources = load_sources()
    fixed = []
    for src, g in preds.groupby("source", sort=False):
        m = pd.DataFrame({"header": g["header"], "predicted": g["cascade"], "decided_by": ""})
        fixed.append(resolve_price_conflict(m, sources[src])["predicted"])
    preds["system"] = pd.concat(fixed)
    results.append({"method": "SYSTEM: cascade + price/MRP value rule",
                    "accuracy": acc(preds["system"]), "llm_calls": int((~hits).sum())})
    res = pd.DataFrame(results)
    res.attrs["llm_report"] = llm_report
    return res, preds


def evaluate_selection(gold: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    sources = load_sources()
    rows = []
    for src, g in gold.groupby("source"):
        prof = select_columns(sources[src][g["header"].tolist()])
        prof["source"] = src
        rows.append(prof)
    prof = pd.concat(rows, ignore_index=True)
    merged = prof.merge(gold[["source", "header", "cluster_text"]],
                        left_on=["source", "column"], right_on=["source", "header"])
    tp = int(((merged["selected"]) & (merged["cluster_text"] == 1)).sum())
    fp = int(((merged["selected"]) & (merged["cluster_text"] == 0)).sum())
    fn = int(((~merged["selected"]) & (merged["cluster_text"] == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}, merged


def build_canonical_tables(threshold: float = 0.6) -> dict[str, pd.DataFrame]:
    """Map four differently shaped files with the hybrid method and save them."""
    tables = {"bigbasket": data_io.load_bigbasket(keep_index=True), "flipkart": data_io.load_flipkart(),
              "off_india": data_io.load_off_india(), "abt": data_io.load_abt_buy()["left"],
              "buy": data_io.load_abt_buy()["right"]}
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    result, mappings = {}, []
    for name, df in tables.items():
        headers = list(df.columns)
        samples = [sample_values(df[h]) for h in headers]
        mapping = resolve_price_conflict(map_columns(headers, samples, threshold=threshold), df)
        mapping.insert(0, "source", name)
        mappings.append(mapping)
        canon = to_canonical(df, mapping, name)
        canon.to_parquet(out_dir / f"{name}_canonical.parquet", index=False)
        result[name] = canon
    pd.concat(mappings).to_csv(ROOT / "reports" / "phase1_mappings_used.csv", index=False)
    return result


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    res, preds = evaluate()
    print(res.to_string(index=False))
    print(res.attrs["llm_report"])
    res.to_csv(ROOT / "reports" / "phase1_mapping_accuracy.csv", index=False)
    preds.to_csv(ROOT / "reports" / "phase1_mapping_predictions.csv", index=False)

    gold = gold_with_samples()
    sel_metrics, sel = evaluate_selection(gold)
    print("column selection:", sel_metrics)
    sel.drop(columns=["header"]).to_csv(ROOT / "reports" / "phase1_column_selection.csv", index=False)

    tables = build_canonical_tables()
    for name, t in tables.items():
        print(name, t.shape, list(t.columns))
    print(llm.report())
    llm.unload()
