"""Phase 6: the LLM layer. Every call goes through src/llm.py (JSON schema, Pydantic validation,
cache, call counter). The LLM is used only for:
  1. grey-zone adjudication  pairs whose hybrid score is close to the threshold
  2. canonical naming        one standard name per cluster
  3. zero-shot categories    compared with an embedding (cosine to category names) baseline
  4. synthetic data          noisy variants + hard negatives for Phase 7 fine-tuning
  (+ the Hinglish rewrites used in Phase 7)
"""
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src import data_io, llm

ROOT = data_io.ROOT
SEED = 42


# ---------------------------------------------------------------- 1. grey-zone adjudication
class SameProduct(BaseModel):
    same: bool
    reason: str = Field(max_length=200)


ADJ_SYSTEM = ("You decide whether two product listings refer to exactly the same product "
              "(same brand, model, size and variant). Different sizes, colours, pack counts or model "
              "numbers mean different products. Answer only with JSON.")


def adjudicate(a: str, b: str) -> SameProduct | None:
    prompt = f"Listing A: {a}\nListing B: {b}\nAre A and B the same product? Give a short reason."
    return llm.ask_json(prompt, SameProduct, system=ADJ_SYSTEM)


def grey_zone(scores: np.ndarray, threshold: float, low: float, high: float) -> np.ndarray:
    """Pairs whose score lies in [threshold - low, threshold + high) go to the LLM."""
    return (scores >= threshold - low) & (scores < threshold + high)


# ---------------------------------------------------------------- 2. canonical naming
class CanonicalName(BaseModel):
    name: str = Field(max_length=120)


NAME_SYSTEM = ("You write one clean, standard product name for a group of listings of the same product: "
               "Brand + product + variant + size, title case, no marketing words. Answer only with JSON.")


def canonical_name(members: list[str]) -> str | None:
    listing = "\n".join(f"- {m}" for m in members[:10])
    out = llm.ask_json(f"Listings of one product:\n{listing}\nStandard name:", CanonicalName, system=NAME_SYSTEM)
    return out.name if out else None


def medoid_name(members: list[str], emb: np.ndarray) -> str:
    """Baseline: the member closest to all others (no LLM)."""
    return members[int((emb @ emb.T).sum(1).argmax())]


# ---------------------------------------------------------------- 3. zero-shot categorization
BB_CATEGORIES = ["Beauty & Hygiene", "Gourmet & World Food", "Kitchen, Garden & Pets", "Snacks & Branded Foods",
                 "Foodgrains, Oil & Masala", "Cleaning & Household", "Beverages", "Bakery, Cakes & Dairy",
                 "Baby Care", "Fruits & Vegetables", "Eggs, Meat & Fish"]
BBCategory = Literal[tuple(BB_CATEGORIES)]


class CategoryBatch(BaseModel):
    categories: list[BBCategory]


CAT_SYSTEM = "You assign grocery-store products to one category each. Answer only with JSON."


def categorize_llm(names: list[str], batch_size: int = 10) -> list[str | None]:
    """Batches of 10 names per call; a reply with the wrong number of labels counts as invalid."""
    out = []
    cats = "\n".join(f"- {c}" for c in BB_CATEGORIES)
    for k in range(0, len(names), batch_size):
        batch = names[k:k + batch_size]
        listing = "\n".join(f"{i + 1}. {n}" for i, n in enumerate(batch))
        res = llm.ask_json(f"Categories:\n{cats}\n\nProducts:\n{listing}\n\n"
                           f"Return the list of {len(batch)} categories, in the same order.",
                           CategoryBatch, system=CAT_SYSTEM)
        if res is None or len(res.categories) != len(batch):
            out += [None] * len(batch)
        else:
            out += list(res.categories)
    return out


def categorize_embedding(names: list[str], model, labels: list[str] = BB_CATEGORIES) -> list[str]:
    """Baseline: cosine similarity between the name and each category name."""
    a = model.encode(names, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    b = model.encode([f"{c} products" for c in labels], normalize_embeddings=True, show_progress_bar=False)
    return [labels[i] for i in (a @ b.T).argmax(1)]


# Google Product Taxonomy (top level) as a second label set; BigBasket -> Google mapping used as gold


def google_top_levels() -> list[str]:
    return sorted(data_io.load_google_taxonomy()["top_level"].unique())


BB_TO_GOOGLE = {
    "Beauty & Hygiene": "Health & Beauty", "Gourmet & World Food": "Food, Beverages & Tobacco",
    "Kitchen, Garden & Pets": "Home & Garden", "Snacks & Branded Foods": "Food, Beverages & Tobacco",
    "Foodgrains, Oil & Masala": "Food, Beverages & Tobacco", "Cleaning & Household": "Home & Garden",
    "Beverages": "Food, Beverages & Tobacco", "Bakery, Cakes & Dairy": "Food, Beverages & Tobacco",
    "Baby Care": "Baby & Toddler", "Fruits & Vegetables": "Food, Beverages & Tobacco",
    "Eggs, Meat & Fish": "Food, Beverages & Tobacco",
}


def categorize_llm_google(names: list[str], batch_size: int = 10) -> list[str | None]:
    labels = google_top_levels()
    GCat = Literal[tuple(labels)]

    class GoogleBatch(BaseModel):
        categories: list[GCat]

    out = []
    cats = "\n".join(f"- {c}" for c in labels)
    for k in range(0, len(names), batch_size):
        batch = names[k:k + batch_size]
        listing = "\n".join(f"{i + 1}. {n}" for i, n in enumerate(batch))
        res = llm.ask_json(f"Google Product Taxonomy top-level categories:\n{cats}\n\nProducts:\n{listing}\n\n"
                           f"Return the list of {len(batch)} categories, in the same order.",
                           GoogleBatch, system=CAT_SYSTEM)
        out += list(res.categories) if res is not None and len(res.categories) == len(batch) else [None] * len(batch)
    return out


# ---------------------------------------------------------------- 4. synthetic data
class Variants(BaseModel):
    noisy_variants: list[str] = Field(min_length=3, max_length=3)
    hard_negative: str


SYN_SYSTEM = ("You create test data for product matching. Answer only with JSON.")


def synthetic(name: str) -> Variants | None:
    prompt = (f"Product: {name}\n"
              "1) Write 3 noisy listings of EXACTLY this product, as another shop might write it: typos, "
              "abbreviations (pkt, gm, ltr), reordered words, different case or punctuation. Keep brand, "
              "variant and size the same.\n"
              "2) Write 1 hard negative: same brand and similar wording but a DIFFERENT product "
              "(other size, pack count or variant).")
    return llm.ask_json(prompt, Variants, system=SYN_SYSTEM)


# ---------------------------------------------------------------- Hinglish rewrites (used in Phase 7)
class HinglishRewrite(BaseModel):
    english_noisy: str
    hinglish: str


HING_SYSTEM = ("You rewrite Indian grocery product names the way Indian shoppers type them. "
               "Answer only with JSON.")


def hinglish(name: str) -> HinglishRewrite | None:
    prompt = (f"Product: {name}\n"
              "english_noisy: the same product written casually in English (lowercase, abbreviations, "
              "word order changed), keep brand and size.\n"
              "hinglish: the same product in Hinglish: replace the common English food/household words with "
              "their Hindi words in Roman script (e.g. turmeric -> haldi, rice -> chawal, lentils -> dal, "
              "sugar -> cheeni, salt -> namak, flour -> atta, oil -> tel), keep brand and size.")
    return llm.ask_json(prompt, HinglishRewrite, system=HING_SYSTEM)


# ================================================================ Phase 6 runner
# Each task has a "prepare" step (embeddings / scores, no LLM) and an "llm" step (no PyTorch),
# because qwen3:8b and a PyTorch model do not fit in this laptop's RAM at the same time.
P6 = ROOT / "data" / "processed" / "phase6"
OUT = ROOT / "reports"


def _metrics_cls(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    y_pred = [p if p is not None else "INVALID" for p in y_pred]
    return {"accuracy": round(accuracy_score(y_true, y_pred), 3),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3),
            "invalid": sum(p == "INVALID" for p in y_pred)}


def prepare_adjudication(low: float = 0.10, high: float = 0.10) -> None:
    """Grey-zone pairs on the TEST parts of Abt-Buy and WDC (0 % unseen), using the Phase 5
    'hybrid + all rules' settings tuned on validation data."""
    import json
    from src.embed import load_benchmark, get_sbert
    from src.match import SBERT, hybrid_score, wdc_pair_features
    meta = json.loads((OUT / "phase5_meta.json").read_text())
    P6.mkdir(parents=True, exist_ok=True)
    # Abt-Buy
    p = meta["abt_buy"]["params"]["hybrid + all rules"]
    c = pd.read_parquet(ROOT / "data" / "processed" / "phase5_abt_buy_candidates.parquet")
    c = c[~c["is_val"]].copy()
    c["score"] = hybrid_score(c, p["w"], p["brand_penalty"], p["model_penalty"])
    c["pred_rule"] = c["score"] >= p["threshold"]
    c["grey"] = grey_zone(c["score"].values, p["threshold"], low, high)
    left, right, gold = load_benchmark("abt_buy")
    c["name_a"] = c["left_id"].map(dict(zip(left["id"], left["raw"])))
    c["name_b"] = c["right_id"].map(dict(zip(right["id"], right["raw"])))
    test_left = set(left["id"]) - set(pd.read_parquet(ROOT / "data" / "processed" /
                                                      "phase5_abt_buy_candidates.parquet").query("is_val")["left_id"])
    c["n_gold_test"] = sum(len(v) for k, v in gold.items() if k in test_left)
    c.to_parquet(P6 / "adj_abt_buy.parquet")
    # WDC
    p = meta["wdc_pairs"]["hybrid + all rules"]
    t = data_io.load_wdc_products("pair", 80, "gs", unseen=0)
    f = wdc_pair_features(t, get_sbert(SBERT))
    f["score"] = hybrid_score(f, p["w"], p["brand_penalty"], p["model_penalty"])
    f["pred_rule"] = f["score"] >= p["threshold"]
    f["grey"] = grey_zone(f["score"].values, p["threshold"], low, high)
    f["label"] = t["label"].values.astype(bool)
    f["name_a"], f["name_b"] = t["title_left"].values, t["title_right"].values
    f.to_parquet(P6 / "adj_wdc.parquet")
    print({"abt_buy_grey": int(c["grey"].sum()), "abt_buy_test_candidates": len(c),
           "wdc_grey": int(f["grey"].sum()), "wdc_test_pairs": len(f)})


def run_adjudication() -> pd.DataFrame:
    from src.match import prf
    rows = []
    for name in ("abt_buy", "wdc"):
        d = pd.read_parquet(P6 / f"adj_{name}.parquet").reset_index(drop=True)
        before, t0 = llm.STATS["calls"], llm.STATS["seconds"]
        decisions = {k: adjudicate(d["name_a"].iat[k], d["name_b"].iat[k]) for k in np.where(d["grey"].values)[0]}
        d["pred_llm"] = d["pred_rule"]
        for k, r in decisions.items():
            if r is not None:
                d.loc[k, "pred_llm"] = r.same
        d["llm_reason"] = [decisions[k].reason if decisions.get(k) else "" for k in range(len(d))]
        d.to_parquet(P6 / f"adj_{name}_done.parquet")
        if name == "abt_buy":
            key = list(zip(d["left_id"], d["right_id"]))
            gold = {k for k, l in zip(key, d["label"]) if l}
            n_missing = int(d["n_gold_test"].iat[0]) - len(gold)     # lost by blocking: always FN
            gold |= {("missing", i) for i in range(n_missing)}
            pr = lambda col: prf({k for k, v in zip(key, d[col]) if v}, gold)
        else:
            pr = lambda col: prf(set(np.where(d[col])[0]), set(np.where(d["label"])[0]))
        grey = d["grey"].values
        rows.append({"dataset": name, "pairs": len(d), "grey_zone_pairs": int(grey.sum()),
                     "llm_calls": llm.STATS["calls"] - before, "llm_seconds": round(llm.STATS["seconds"] - t0, 1),
                     **{f"before_{k}": v for k, v in pr("pred_rule").items()},
                     **{f"after_{k}": v for k, v in pr("pred_llm").items()},
                     "grey_acc_rule": round(float((d["pred_rule"][grey] == d["label"][grey]).mean()), 3),
                     "grey_acc_llm": round(float((d["pred_llm"][grey] == d["label"][grey]).mean()), 3)})
        print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "phase6_adjudication.csv", index=False)
    return res


def prepare_names(n_clusters: int = 100) -> None:
    from src.embed import get_sbert
    from src.match import SBERT
    bb = pd.read_parquet(ROOT / "data" / "processed" / "bigbasket_clusters.parquet")
    sizes = bb["cluster_id"].value_counts()
    multi = sizes[sizes > 1].index.to_series().sample(min(n_clusters, int((sizes > 1).sum())), random_state=SEED)
    model = get_sbert(SBERT)
    rows = []
    for cid in multi:
        m = bb[bb["cluster_id"] == cid]
        names = (m["brand"].fillna("") + " " + m["product_name"]).str.strip().tolist()
        emb = model.encode(names, normalize_embeddings=True, show_progress_bar=False)
        rows.append({"cluster_id": int(cid), "size": len(m), "members": names, "brand": m["brand"].iat[0],
                     "medoid_name": medoid_name(names, emb)})
    P6.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_json(P6 / "names_input.json", orient="records", indent=1, force_ascii=False)


def run_names() -> pd.DataFrame:
    d = pd.read_json(P6 / "names_input.json")
    d["llm_name"] = [canonical_name(m) for m in d["members"]]
    has = lambda name, brand: isinstance(name, str) and str(brand).lower() in name.lower()
    d["llm_has_brand"] = [has(n, b) for n, b in zip(d["llm_name"], d["brand"])]
    d["medoid_has_brand"] = [has(n, b) for n, b in zip(d["medoid_name"], d["brand"])]
    d.to_json(OUT / "phase6_canonical_names.json", orient="records", indent=1, force_ascii=False)
    return d


def category_sample(per_class: int = 50) -> pd.DataFrame:
    bb = data_io.load_bigbasket()
    s = pd.concat([g.sample(min(per_class, len(g)), random_state=SEED) for _, g in bb.groupby("category")])
    s = s.reset_index(drop=True)
    s["text"] = (s["brand"].fillna("") + " " + s["product"]).str.strip()
    return s


def prepare_categories() -> None:
    from src.embed import get_sbert
    s = category_sample()
    for key, name in (("minilm", "sentence-transformers/all-MiniLM-L6-v2"), ("bge", "BAAI/bge-small-en-v1.5")):
        s[f"emb_{key}"] = categorize_embedding(s["text"].tolist(), get_sbert(name))
    P6.mkdir(parents=True, exist_ok=True)
    s.to_parquet(P6 / "categories_input.parquet")


def run_categories() -> pd.DataFrame:
    s = pd.read_parquet(P6 / "categories_input.parquet")
    before = llm.STATS["calls"]
    s["llm_bb"] = categorize_llm(s["text"].tolist())
    calls_bb, before = llm.STATS["calls"] - before, llm.STATS["calls"]
    s["llm_google"] = categorize_llm_google(s["text"].tolist())
    calls_g = llm.STATS["calls"] - before
    s["gold_google"] = s["category"].map(BB_TO_GOOGLE)
    rows = [{"label set": "BigBasket (11)", "method": "embedding cosine, MiniLM",
             **_metrics_cls(s["category"], s["emb_minilm"]), "llm_calls": 0},
            {"label set": "BigBasket (11)", "method": "embedding cosine, BGE-small",
             **_metrics_cls(s["category"], s["emb_bge"]), "llm_calls": 0},
            {"label set": "BigBasket (11)", "method": "LLM zero-shot (qwen3:8b)",
             **_metrics_cls(s["category"], s["llm_bb"]), "llm_calls": calls_bb},
            {"label set": "Google top-level (21)", "method": "LLM zero-shot (qwen3:8b)",
             **_metrics_cls(s["gold_google"], s["llm_google"]), "llm_calls": calls_g}]
    s.to_parquet(OUT / "phase6_categories_preds.parquet")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "phase6_categorization.csv", index=False)
    return res


def synthetic_names(n: int = 200, offset: int = 0) -> list[str]:
    bb = data_io.load_bigbasket().drop_duplicates("product").sample(frac=1, random_state=SEED + 1)
    names = (bb["brand"].fillna("") + " " + bb["product"]).str.strip().tolist()
    return names[offset:offset + n]


def run_synthetic(n: int = 200) -> pd.DataFrame:
    rows = []
    for name in synthetic_names(n):
        v = synthetic(name)
        if v is not None:
            rows.append({"anchor": name, "variants": v.noisy_variants, "hard_negative": v.hard_negative})
    d = pd.DataFrame(rows)
    d.to_json(ROOT / "data" / "processed" / "synthetic_pairs.json", orient="records", indent=1, force_ascii=False)
    return d


def run_hinglish(n_test: int = 200) -> pd.DataFrame:
    """Test names are disjoint from the synthetic-data names (different offset)."""
    rows = []
    for name in synthetic_names(n_test, offset=1000):
        h = hinglish(name)
        if h is not None:
            rows.append({"original": name, "english_noisy": h.english_noisy, "hinglish": h.hinglish,
                         "verified_by_student": "no"})
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "data" / "gold" / "hinglish_test.csv", index=False)
    return d


if __name__ == "__main__":
    import json
    import sys
    import time
    pd.set_option("display.width", 220)
    step = sys.argv[1]
    t0 = time.perf_counter()
    if step == "prepare":
        prepare_adjudication()
        prepare_names()
        prepare_categories()
    elif step == "adjudicate":
        print(run_adjudication().to_string(index=False))
    elif step == "names":
        d = run_names()
        print(d[["size", "medoid_name", "llm_name"]].head(20).to_string(index=False))
        print("brand kept: llm", d["llm_has_brand"].mean(), "medoid", d["medoid_has_brand"].mean())
    elif step == "categories":
        print(run_categories().to_string(index=False))
    elif step == "synthetic":
        print(len(run_synthetic()), "synthetic anchors")
    elif step == "hinglish":
        print(len(run_hinglish()), "hinglish test items")
    if step != "prepare":
        print(llm.report(), f"wall {time.perf_counter() - t0:.0f}s")
        stats_path = OUT / "phase6_llm_usage.json"
        usage = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        usage[step] = {**llm.STATS, "wall_seconds": round(time.perf_counter() - t0, 1)}
        stats_path.write_text(json.dumps(usage, indent=1))
        llm.unload()
